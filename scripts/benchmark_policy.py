#!/usr/bin/env python3
"""Paired end-to-end benchmark of an installed lplearn policy-plot CLI."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import numpy as np


def atomic_json(path: Path, value) -> None:
    """Publish a complete JSON document with an atomic rename."""
    data = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_bytes(path: Path, data: bytes) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def document_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


@contextlib.contextmanager
def run_lock(root: Path):
    """Use an OS-held lock, so a crashed process leaves no stale ownership."""
    handle = (root / ".run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


def schedule(cases, repetitions, seed):
    rng = random.Random(seed)
    order = []
    jobs = []
    for rep in range(repetitions):
        if rep % len(cases) == 0:
            order = cases.copy()
            rng.shuffle(order)
        shift = rep % len(order)
        jobs.extend((case, rep) for case in order[shift:] + order[:shift])
    return jobs


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(cwd: Path) -> dict:
    """Record the containing checkout and a hash of candidate source files."""
    excluded = {".git", ".venv", "target", "__pycache__", "node_modules"}
    hasher = hashlib.sha256()
    files = 0
    for path in sorted(cwd.rglob("*")):
        if not path.is_file() or path.is_symlink() or any(x in excluded for x in path.relative_to(cwd).parts):
            continue
        if path.suffix not in {".py", ".rs", ".toml", ".lock"}:
            continue
        hasher.update(str(path.relative_to(cwd)).encode() + b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
        files += 1
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--short", "--untracked-files=normal"], cwd=cwd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, None
    return {"git_commit": commit, "git_status": status, "source_tree_sha256": hasher.hexdigest(), "source_file_count": files}


def environment_identity(python: str, cwd: Path | None = None) -> dict:
    program = '''import importlib.metadata as m, json, platform, sys
distributions = sorted((d.metadata.get("Name", ""), d.version) for d in m.distributions())
try:
 from lplearn.rust import _executable
 executable = str(_executable())
except Exception:
 executable = None
print(json.dumps({"python_executable": sys.executable, "python_prefix": sys.prefix, "python_version": sys.version, "platform": platform.platform(), "distributions": distributions, "rust_binary": executable}))'''
    try:
        proc = subprocess.run([python, "-c", program], cwd=cwd, capture_output=True, text=True, check=True, timeout=30)
        return json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"error": str(exc), "rust_binary": None}


def _windows_physical_memory() -> int | None:
    """Read physical memory via the Windows MEMORYSTATUSEX ABI."""
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.DWORD),
            ("load", wintypes.DWORD),
            ("total_physical", ctypes.c_uint64),
            ("available_physical", ctypes.c_uint64),
            ("total_page_file", ctypes.c_uint64),
            ("available_page_file", ctypes.c_uint64),
            ("total_virtual", ctypes.c_uint64),
            ("available_virtual", ctypes.c_uint64),
            ("reserved", ctypes.c_uint64),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    read_status = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    read_status.argtypes = [ctypes.POINTER(MemoryStatus)]
    read_status.restype = wintypes.BOOL
    if read_status(ctypes.byref(status)):
        return int(status.total_physical) or None
    return None


def hardware_identity() -> dict:
    cpu = platform.processor() or None
    memory = None
    if sys.platform == "darwin":
        for key, field in (("machdep.cpu.brand_string", "cpu"), ("hw.memsize", "memory")):
            try:
                value = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
                if field == "cpu": cpu = value
                else: memory = int(value)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
    if sys.platform == "win32":
        try:
            memory = _windows_physical_memory()
        except (OSError, AttributeError):
            pass
    elif memory is None:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                memory = pages * page_size
        except (ValueError, OSError, AttributeError):
            pass
    if (not cpu or cpu.lower() in {"arm", "unknown"}) and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "memory_bytes": memory}


def file_identity(path: str | None) -> dict | None:
    if not path:
        return None
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        return None
    stat = file.stat()
    return {"path": str(file), "sha256": digest(file), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def lock_matches_environment(lock: Path, distributions: list) -> tuple[bool, str]:
    """Check an exhaustive, exact name==version lock against installed packages.

    This confirms installed names and versions, not package file integrity or
    that the lock was used for installation. Unsupported pip syntax fails closed.
    """
    def normalize(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    expected = {}
    try:
        for number, line in enumerate(lock.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)", line)
            if not match:
                return False, f"unsupported lock entry on line {number}"
            name, version = normalize(match[1]), match[2]
            if name in expected:
                return False, f"duplicate package {name} in lock"
            expected[name] = version
        installed = {}
        for name, version in distributions:
            key = normalize(name)
            if not key or key in installed:
                return False, f"duplicate or unnamed installed distribution: {name!r}"
            installed[key] = version
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        return False, f"cannot compare lock: {exc}"
    if not expected:
        return False, "lock contains no distributions"
    missing = sorted(expected.keys() - installed.keys())
    extra = sorted(installed.keys() - expected.keys())
    different = sorted(k for k in expected.keys() & installed.keys() if expected[k] != installed[k])
    if missing or extra or different:
        return False, f"lock mismatch: missing={missing}, extra={extra}, different={different}"
    return True, "installed names and versions match lock"


def cli_environment(command: list[str], cwd: Path | None = None) -> dict:
    """Identify the interpreter used by a direct Python command or script.

    Native launchers and shell wrappers are intentionally unresolved: their
    interpreter cannot be established from the command alone.
    """
    path = resolve_command(command[0], cwd)
    try:
        with open(path, "rb") as executable:
            shebang = executable.readline(4096)
    except OSError as exc:
        return {"error": str(exc)}
    interpreter = path
    if shebang.startswith(b"#!"):
        try:
            parts = shlex.split(shebang[2:].decode("utf-8").strip())
        except (UnicodeError, ValueError) as exc:
            return {"error": f"invalid shebang: {exc}"}
        if not parts:
            return {"error": "empty shebang"}
        if Path(parts[0]).name == "env":
            if len(parts) != 2:
                return {"error": "unsupported env shebang"}
            interpreter = shutil.which(parts[1])
        else:
            interpreter = parts[0]
        if not interpreter:
            return {"error": "shebang interpreter not found"}
    # Native launchers and arbitrary scripts must not be run with '-c'.
    if "python" not in Path(interpreter).name.lower():
        return {"error": "cannot establish the Python interpreter for CLI"}
    return environment_identity(interpreter, cwd)


def resolve_command(command: str, cwd: Path | None) -> str:
    if os.path.dirname(command):
        path = Path(command)
        return str(path if path.is_absolute() else (cwd or Path.cwd()) / path)
    return shutil.which(command) or command


def wheel_binary_matches(wheel: str | None, binary: dict | None) -> bool:
    if not wheel or not binary:
        return False
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if Path(n).name == Path(binary["path"]).name]
            return len(names) == 1 and hashlib.sha256(archive.read(names[0])).hexdigest() == binary["sha256"]
    except (OSError, zipfile.BadZipFile):
        return False


def wheel_python_matches(python: str, wheel: str | None, cwd: Path | None = None) -> bool:
    """Verify installed Python sources against the wheel using its interpreter."""
    if not wheel:
        return False
    program = '''import hashlib, importlib.metadata as m, importlib.util as u, pathlib, sys, zipfile
try:
 dist = m.distribution("lplearn")
 with zipfile.ZipFile(sys.argv[1]) as archive:
  names = [n for n in archive.namelist() if n.endswith(".py") and n.startswith(("lplearn/", "lp_learner/"))]
  ok = bool(names) and all(hashlib.sha256(archive.read(n)).digest() == hashlib.sha256(dist.locate_file(n).read_bytes()).digest() for n in names)
  roots = {n.partition("/")[0] for n in names}
  ok = ok and all(u.find_spec(r) is not None and pathlib.Path(u.find_spec(r).origin).resolve() == dist.locate_file(r + "/__init__.py").resolve() for r in roots)
 print("yes" if ok else "no")
except Exception:
 print("no")'''
    try:
        result = subprocess.run([python, "-c", program, wheel], cwd=cwd, capture_output=True, text=True, check=True, timeout=30)
        return result.stdout.strip() == "yes"
    except (OSError, subprocess.SubprocessError):
        return False


def cache_identity(directory: Path | None) -> str | None:
    if directory is None:
        return None
    hasher = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            hasher.update(str(path.relative_to(directory)).encode() + b"\0")
            hasher.update(bytes.fromhex(digest(path)))
    return hasher.hexdigest()


def parse_case(value: str) -> dict:
    parts = value.split(":")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("case must be NAME:METHOD:BACKEND")
    name, method, backend = parts
    if not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("case name must contain only letters, digits, '-' or '_'")
    return {"name": name, "method": method, "backend": backend}


class UncertainWorkerTreeError(RuntimeError):
    """The CLI stopped but one or more descendants might still be running."""


def stop_timed_out_worker(worker, *, is_windows):
    """Stop a timed-out worker or abort when its descendants cannot be verified."""
    if is_windows:
        # Killing the CLI does not guarantee that its Rust child has stopped.
        # Keep the attempt uncommitted and refuse to run any subsequent job.
        try:
            worker.kill()
        except OSError:
            pass
        raise UncertainWorkerTreeError(
            "Windows CLI timed out; descendant solver processes may still be running. "
            "Inspect and stop them before starting a new benchmark run."
        )
    os.killpg(worker.pid, signal.SIGKILL)
    return worker.communicate()


def verify_recovery(manifest, original, output):
    """Independently verify the projection and derived policy against the model."""
    from importlib import metadata
    import lplearn
    from lplearn.core import QueueParams, queue_wcmdp
    from lplearn.core.policy import conditional_policy, fluid_evaluation
    from lplearn.workflow import occupation_feasibility

    installed = Path(metadata.distribution("lplearn").locate_file("lplearn/__init__.py")).resolve()
    if Path(lplearn.__file__).resolve() != installed:
        raise ValueError("recovery verifier did not load the installed lplearn package")
    fields = dict(manifest["config"]["model"])
    if fields.get("m0") is not None:
        fields["m0"] = tuple(fields["m0"])
    model = queue_wcmdp(QueueParams(**fields))
    recovery = manifest["recovery"]
    if (recovery["method"] != "euclidean_projection_lp_relaxation"
            or recovery["solver"] != "CLARABEL"
            or recovery["status"] not in ("optimal", "optimal_inaccurate")):
        raise ValueError("unexpected recovery method, solver or status")
    for field in ("seconds", "distance_l2"):
        value = recovery[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid recovery {field}")
    recovered_path = output.with_name(output.stem + ".recovered.npy")
    policy_path = output.with_name(output.stem + ".policy.npy")
    if (str(recovered_path) != recovery["occupation_path"] or str(policy_path) != recovery["conditional_policy_path"]
            or digest(recovered_path) != recovery["occupation_sha256"]
            or digest(policy_path) != recovery["conditional_policy_sha256"]):
        raise ValueError("recovery artifact path or hash mismatch")
    recovered = np.load(recovered_path, allow_pickle=False)
    policy = np.load(policy_path, allow_pickle=False)
    if recovered.shape != original.shape or policy.shape != original.shape or not np.isfinite(recovered).all() or not np.isfinite(policy).all():
        raise ValueError("invalid recovered occupation or conditional policy")
    if policy.min() < 0 or not np.allclose(policy.sum(axis=2), 1.0, rtol=0, atol=1e-12):
        raise ValueError("invalid conditional policy probabilities")
    if not np.allclose(policy, conditional_policy(recovered), rtol=1e-12, atol=1e-12):
        raise ValueError("conditional policy does not correspond to recovered occupation")
    original_check = occupation_feasibility(original, model)
    recovered_check = occupation_feasibility(recovered, model)
    policy_check = occupation_feasibility(fluid_evaluation(model, policy)["y"], model)
    if not recovered_check["feasible"] or not policy_check["feasible"]:
        raise ValueError("recovered occupation or evaluated policy violates LP constraints")
    for key, checked in (("original_feasibility", original_check), ("recovered_feasibility", recovered_check),
                         ("conditional_policy_feasibility", policy_check)):
        reported = recovery[key]
        if reported != checked:
            raise ValueError(f"incorrect {key} report")
    if manifest["occupation_feasibility"] != original_check:
        raise ValueError("incorrect original occupation feasibility report")
    if not math.isclose(recovery["distance_l2"], float(np.linalg.norm(recovered - original)), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("incorrect projection distance")
    return recovered_path, policy_path


def run_one(command, cwd, frozen_config, case, repetition, directory, timeout, environment, process_started=None, rust_timeout=None, allow_infeasible=False, recover_learning=False):
    output = directory / f"{case['name']}-{repetition:03}.png"
    args = [*command, "plot-policy", "--config", str(frozen_config),
            "--method", case["method"], "--backend", case["backend"], "--output", str(output)]
    if rust_timeout is not None:
        args += ["--rust-timeout", str(rust_timeout)]
    if allow_infeasible:
        args.append("--allow-infeasible")
    recovered = bool(recover_learning and case["method"] in {"gda", "sgda"})
    if recovered:
        args.append("--recover")
    start = time.perf_counter()
    error = None
    try:
        with subprocess.Popen(args, cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=(os.name != "nt")) as worker:
            if process_started:
                try:
                    process_started(worker.pid)
                except Exception:
                    worker.terminate()
                    worker.wait()
                    raise
            try:
                stdout, stderr = worker.communicate(timeout=timeout)
                code = worker.returncode
            except subprocess.TimeoutExpired:
                stdout, stderr = stop_timed_out_worker(worker, is_windows=(os.name == "nt"))
                code, error = None, f"timeout after {timeout}s"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        code, error = None, f"timeout after {timeout}s"
    except OSError as exc:
        code, stdout, stderr, error = None, "", "", str(exc)
    wall = time.perf_counter() - start
    record = {"case": case, "repetition": repetition, "command": args, "exit_code": code,
              "wall_seconds": wall, "stdout": stdout, "stderr": stderr, "error": error,
              "recovery_requested": recovered}
    if code != 0:
        return record
    manifest_path = output.with_suffix(".json")
    array_path = output.with_suffix(".npy")
    try:
        manifest = json.loads(manifest_path.read_text())
        array = np.load(array_path, allow_pickle=False)
        if manifest["method"] != case["method"] or manifest["backend"] != case["backend"]:
            raise ValueError("manifest method/backend differs from requested case")
        if manifest["config_sha256"] != digest(frozen_config):
            raise ValueError("manifest configuration hash mismatch")
        if manifest["plot_sha256"] != digest(output) or manifest["occupation_sha256"] != digest(array_path):
            raise ValueError("artifact hash mismatch")
        if array.ndim != 3 or not array.size or not np.isfinite(array).all():
            raise ValueError("occupation array must be non-empty, finite, and three-dimensional")
        if not isinstance(manifest["config"], dict):
            raise ValueError("resolved configuration is missing")
        if recover_learning:
            from lplearn.workflow import load_config
            expected_config = json.loads(json.dumps(load_config(frozen_config), allow_nan=False))
            if manifest["config"] != expected_config:
                raise ValueError("resolved configuration differs from selected TOML")
        for field in ("solve_seconds", "plot_seconds"):
            value = manifest[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field}")
        feasibility = manifest["occupation_feasibility"]
        if not isinstance(feasibility, dict) or type(feasibility.get("feasible")) is not bool:
            raise ValueError("missing occupation feasibility status")
        if recovered:
            if manifest.get("plot_kind") != "recovered_occupation":
                raise ValueError("recovered plot was not labeled")
            recovered_path, policy_path = verify_recovery(manifest, array, output)
            record.update(recovered_array=str(recovered_path), policy_array=str(policy_path))
        else:
            expected_kind = "occupation" if feasibility["feasible"] else "infeasible_iterate_diagnostic"
            if manifest.get("plot_kind") != expected_kind or (not feasibility["feasible"] and not allow_infeasible):
                raise ValueError("infeasible plot was not explicitly authorized and labeled")
            if "recovery" in manifest:
                raise ValueError("unexpected recovery in direct case")
        if case["method"] in {"gda", "sgda"}:
            for field in ("seed", "iterations", "samples_per_pair", "averaged_iterations"):
                if type(manifest[field]) is not int or (field != "seed" and manifest[field] < 0):
                    raise ValueError(f"invalid {field}")
        record.update(manifest=manifest, array=str(array_path), artifact_verified=True)
    except (OSError, EOFError, ValueError, KeyError, TypeError, ImportError, AttributeError, OverflowError) as exc:
        record["error"] = f"invalid/missing artifact: {exc}"
    return record


def orphan_worker_alive(folder: Path) -> bool:
    """Fail closed if an abandoned attempt might still own a running worker."""
    try:
        state = json.loads((folder / ".worker.json").read_text())
        pid = state["pid"]
        if type(pid) is not int or pid <= 0 or state["status"] not in {"started", "finished"}:
            raise ValueError("invalid worker state")
        if state["status"] == "finished":
            return False
        if os.name == "nt":
            return True  # Descendant processes cannot be ruled out by checking one PID.
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"orphan worker state unknown for {folder}: {exc}") from exc


def summarize(cases, records, atol, rtol, bootstrap, seed, recovery_atol=2e-7, recovery_rtol=2e-7):
    summary = {"cases": {}, "paired": []}
    for case in cases:
        rows = [r for r in records if r["case"] == case and r.get("artifact_verified")]
        summary["cases"][case["name"]] = {
            "requested": sum(r["case"] == case for r in records), "successful": len(rows),
            "feasible": sum(r["manifest"]["occupation_feasibility"]["feasible"] for r in rows),
            "raw_occupation_feasible": sum(r["manifest"]["occupation_feasibility"]["feasible"] for r in rows),
            "plotted_relaxation_feasible": sum(r["manifest"].get("plot_kind") in {"occupation", "recovered_occupation"} for r in rows),
            "infeasible_diagnostics": sum(r["manifest"].get("plot_kind") == "infeasible_iterate_diagnostic" for r in rows),
            "recovered": sum(bool(r.get("recovery_requested")) for r in rows),
            "median_wall_seconds": statistics.median(r["wall_seconds"] for r in rows) if rows else None,
            "median_solve_seconds": statistics.median(r["manifest"]["solve_seconds"] for r in rows) if rows else None,
            "median_plot_seconds": statistics.median(r["manifest"]["plot_seconds"] for r in rows) if rows else None,
            "median_projection_solve_seconds": statistics.median(r["manifest"]["recovery"]["seconds"] for r in rows if r.get("recovery_requested")) if any(r.get("recovery_requested") for r in rows) else None,
        }
    rng = random.Random(seed)
    for i, base in enumerate(cases):
        for candidate in cases[i + 1:]:
            if base["method"] != candidate["method"] or base["backend"] == candidate["backend"]:
                continue
            base_rows = {r["repetition"]: r for r in records if r["case"] == base and r.get("artifact_verified")}
            candidate_rows = {r["repetition"]: r for r in records if r["case"] == candidate and r.get("artifact_verified")}
            pairs = []
            for rep in sorted(base_rows.keys() & candidate_rows.keys()):
                a, b = base_rows[rep], candidate_rows[rep]
                x = np.load(a["array"], allow_pickle=False)
                y = np.load(b["array"], allow_pickle=False)
                same_shape = x.shape == y.shape
                difference = float(np.max(np.abs(x - y))) if same_shape and x.size else (0.0 if same_shape else None)
                numeric = bool(same_shape and np.allclose(x, y, atol=atol, rtol=rtol))
                settings = ("config", "seed", "iterations", "samples_per_pair", "averaged_iterations")
                same_settings = all(key in a["manifest"] and key in b["manifest"]
                                    and a["manifest"][key] == b["manifest"][key] for key in settings)
                same_recovery = a.get("recovery_requested", False) == b.get("recovery_requested", False)
                recovered_parity = policy_parity = None
                recovered_difference = policy_difference = None
                if same_recovery and a.get("recovery_requested", False):
                    xr, yr = np.load(a["recovered_array"], allow_pickle=False), np.load(b["recovered_array"], allow_pickle=False)
                    xp, yp = np.load(a["policy_array"], allow_pickle=False), np.load(b["policy_array"], allow_pickle=False)
                    recovered_parity = bool(xr.shape == yr.shape and np.allclose(xr, yr, atol=recovery_atol, rtol=recovery_rtol))
                    policy_parity = bool(xp.shape == yp.shape and np.allclose(xp, yp, atol=recovery_atol, rtol=recovery_rtol))
                    recovered_difference = float(np.max(np.abs(xr - yr))) if xr.shape == yr.shape and xr.size else None
                    policy_difference = float(np.max(np.abs(xp - yp))) if xp.shape == yp.shape and xp.size else None
                    same_recovery = all(a["manifest"]["recovery"][key] == b["manifest"]["recovery"][key]
                                        for key in ("method", "solver"))
                full_parity = numeric and same_settings and same_recovery and recovered_parity is not False and policy_parity is not False
                pairs.append({"repetition": rep, "parity": full_parity, "numeric_parity": numeric,
                              "same_effective_settings": same_settings, "max_absolute_difference": difference,
                              "same_recovery": same_recovery, "recovered_occupation_parity": recovered_parity,
                              "conditional_policy_parity": policy_parity, "recovered_max_absolute_difference": recovered_difference,
                              "policy_max_absolute_difference": policy_difference,
                              "wall_ratio_baseline_over_candidate": a["wall_seconds"] / b["wall_seconds"]})
            entry = {"baseline": base["name"], "candidate": candidate["name"], "method": base["method"],
                     "successful_pairs": len(pairs), "all_pairs_match": bool(pairs) and all(p["parity"] for p in pairs),
                     "atol": atol, "rtol": rtol, "recovery_atol": recovery_atol, "recovery_rtol": recovery_rtol,
                     "pairs": pairs, "ratio_estimate": None}
            # Only report a speed ratio if all requested pairs completed and agreed.
            requested = sum(r["case"] == base for r in records)
            if len(pairs) >= 5 and len(pairs) == requested and entry["all_pairs_match"]:
                ratios = [p["wall_ratio_baseline_over_candidate"] for p in pairs]
                medians = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(bootstrap))
                entry["ratio_estimate"] = {"median": statistics.median(ratios),
                    "bootstrap_95_percentile": [medians[int(.025 * (bootstrap - 1))], medians[int(.975 * (bootstrap - 1))]],
                    "meaning": "baseline wall time / candidate wall time"}
            summary["paired"].append(entry)
    return summary


def committed_shards(root: Path, run_id: str, jobs: list) -> dict:
    """Reload and verify all committed shards and their immutable artifacts."""
    expected = {f"{case['name']}-{rep:05}": (case, rep) for case, rep in jobs}
    result = {}
    for path in sorted((root / "shards").glob("*.json")):
        shard = json.loads(path.read_text())
        payload = shard.get("payload")
        if not isinstance(payload, dict) or shard.get("sha256") != document_hash(payload):
            raise ValueError(f"invalid shard hash: {path}")
        job_id = payload.get("job_id")
        if path.stem != job_id or job_id not in expected or payload.get("run_id") != run_id or payload.get("row_count") != 1:
            raise ValueError(f"unexpected shard identity: {path}")
        case, rep = expected[job_id]
        record = payload.get("record", {})
        if record.get("case") != case or record.get("repetition") != rep:
            raise ValueError(f"shard job mismatch: {path}")
        attempt = payload.get("attempt")
        if not isinstance(attempt, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", attempt):
            raise ValueError(f"invalid attempt identity: {path}")
        folder = root / "attempts" / attempt
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"missing attempt: {folder}")
        entries = list(folder.iterdir())
        if any(not p.is_file() or p.is_symlink() for p in entries):
            raise ValueError(f"unexpected attempt artifact: {folder}")
        actual = {p.name: digest(p) for p in entries}
        if actual != payload.get("artifacts"):
            raise ValueError(f"changed attempt artifacts: {folder}")
        if record.get("artifact_verified"):
            stem = f"{case['name']}-{rep:03}"
            array = folder / f"{stem}.npy"
            manifest = folder / f"{stem}.json"
            plot = folder / f"{stem}.png"
            if str(array) != record.get("array") or json.loads(manifest.read_text()) != record.get("manifest"):
                raise ValueError(f"changed manifest or array location: {path}")
            if digest(array) != record["manifest"]["occupation_sha256"] or digest(plot) != record["manifest"]["plot_sha256"]:
                raise ValueError(f"changed output hashes: {path}")
        result[job_id] = payload
    return result


def progress_writer(root: Path, total: int, prior_elapsed: float, known_orphans: list):
    start = time.monotonic()
    mutex = threading.Lock()

    def publish(status, phase, completed, active=None):
        with mutex:
            elapsed = prior_elapsed + time.monotonic() - start
            rate = completed / elapsed if elapsed > 0 else 0.0
            try:
                load = list(os.getloadavg())
            except (AttributeError, OSError):
                load = None
            try:
                import resource
                peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            except ImportError:
                peak_rss = None
            post_phases = {"execution": 0, "collection": 0, "aggregation": 1,
                           "terminal_verification": 2, "complete": 3, "failed": 3}
            fraction = (completed + post_phases.get(phase, 0)) / (total + 3)
            value = {"status": status, "phase": phase, "active_item": active,
                     "completed": completed, "total": total,
                     "fraction": fraction,
                     "elapsed_seconds": elapsed, "units_per_second": rate,
                     "eta_seconds": (total - completed) / rate if rate > 0 and completed < total else None,
                     "updated_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                     "workers": 1, "load_average": load, "parent_peak_rss_raw": peak_rss,
                     "parent_peak_rss_units": "bytes" if sys.platform == "darwin" else "KiB",
                     "orphaned_attempts": known_orphans}
            atomic_json(root / "progress.json", value)
    return publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help="quoted installed CLI command, e.g. '/path/to/venv/bin/lplearn'")
    parser.add_argument("--cwd", required=True, help="working directory for the CLI")
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", action="append", type=parse_case, required=True, help="NAME:METHOD:BACKEND; repeat")
    parser.add_argument("--output-dir", required=True, help="new directory for immutable config, receipts and plots")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--rust-timeout", type=float, default=240, help="Rust worker timeout in seconds, smaller than --timeout")
    plot_mode = parser.add_mutually_exclusive_group()
    plot_mode.add_argument("--allow-infeasible", action="store_true",
                        help="uniformly allow explicitly labeled infeasible diagnostic plots")
    plot_mode.add_argument("--recover", action="store_true",
                           help="recover learning occupations with QP; leave direct LP unchanged")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-11)
    parser.add_argument("--rtol", type=float, default=2e-11)
    parser.add_argument("--recovery-atol", type=float, default=2e-7,
                        help="absolute tolerance for recovered occupations and policies")
    parser.add_argument("--recovery-rtol", type=float, default=2e-7,
                        help="relative tolerance for recovered occupations and policies")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--mode", choices=("qualification", "definitive"), default="qualification")
    parser.add_argument("--environment-python", default=sys.executable, help="Python belonging to the CLI installation")
    parser.add_argument("--wheel", help="wheel from which CLI was installed")
    parser.add_argument("--rust-binary", help="explicit installed Rust solver path; otherwise resolve using its distribution RECORD")
    parser.add_argument("--lock", help="frozen dependency lock file")
    parser.add_argument("--cpu-model", help="CPU model if operating system information is unavailable")
    parser.add_argument("--numba-cache-state", choices=("per-process-empty", "prewarmed"), required=True)
    parser.add_argument("--numba-cache-dir", help="required pre-populated cache directory for prewarmed runs")
    parser.add_argument("--resume", action="store_true", help="verify and resume an existing incomplete run")
    parser.add_argument("--stop-after", type=int, help="qualification only: interrupt at a job boundary after N new jobs")
    args = parser.parse_args()
    if (args.repetitions < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or args.bootstrap < 100
            or not math.isfinite(args.rust_timeout) or args.rust_timeout <= 0 or args.rust_timeout >= args.timeout
            or not math.isfinite(args.atol) or not math.isfinite(args.rtol) or args.atol < 0 or args.rtol < 0
            or not math.isfinite(args.recovery_atol) or not math.isfinite(args.recovery_rtol)
            or args.recovery_atol < 0 or args.recovery_rtol < 0):
        parser.error("repetitions and timeouts must be positive; rust-timeout must be less than timeout; bootstrap >= 100 and tolerances nonnegative")
    if len({c["name"] for c in args.case}) != len(args.case):
        parser.error("case names must be unique")
    if args.stop_after is not None and (args.stop_after < 1 or args.mode != "qualification"):
        parser.error("--stop-after requires a positive count and qualification mode")
    command = shlex.split(args.command)
    if not command:
        parser.error("command cannot be empty")
    cwd, config, root = Path(args.cwd).resolve(), Path(args.config).resolve(), Path(args.output_dir).resolve()
    environment = environment_identity(args.environment_python, cwd)
    # Independent recovery verification imports the same installed model code
    # as the measured CLI; fail closed if this interpreter is a different env.
    if args.recover and environment_identity(sys.executable, cwd) != environment:
        parser.error("--recover requires this script and CLI to use the same Python environment")
    cli_env = cli_environment(command, cwd)
    env_keys = ("python_prefix", "python_version", "distributions", "rust_binary")
    same_environment = not environment.get("error") and not cli_env.get("error") and all(
        key in environment and key in cli_env and environment[key] == cli_env[key] for key in env_keys)
    hardware = hardware_identity()
    if args.cpu_model:
        hardware["cpu"] = args.cpu_model
        hardware["cpu_source"] = "user-supplied"
    else:
        hardware["cpu_source"] = "operating-system"
    wheel = file_identity(args.wheel)
    rust = file_identity(args.rust_binary or environment.get("rust_binary"))
    cli_executable = file_identity(resolve_command(command[0], cwd))
    env_python = file_identity(shutil.which(args.environment_python) or args.environment_python)
    lock = file_identity(args.lock)
    lock_ok, lock_reason = (lock_matches_environment(Path(lock["path"]), environment.get("distributions", []))
                            if lock else (False, "dependency lock missing"))
    matching_wheel = wheel_binary_matches(wheel["path"] if wheel else None, rust)
    matching_python = wheel_python_matches(args.environment_python, wheel["path"] if wheel else None, cwd)
    cache = Path(args.numba_cache_dir).resolve() if args.numba_cache_dir else None
    if args.numba_cache_state == "prewarmed" and (not cache or not cache.is_dir() or not any(cache.rglob("*.nbc"))):
        parser.error("prewarmed requires --numba-cache-dir containing Numba .nbc cache files")
    if args.mode == "definitive":
        missing = []
        if not wheel: missing.append("wheel")
        if not rust: missing.append("installed Rust binary")
        if not cli_executable or not env_python: missing.append("CLI executable or environment interpreter")
        if not matching_wheel: missing.append("wheel whose Rust binary matches the installed executable")
        if not matching_python: missing.append("wheel whose Python sources match the installed package")
        if not lock: missing.append("dependency lock file")
        if environment.get("error") or not environment.get("distributions"): missing.append("installed dependency inventory")
        if not same_environment: missing.append("CLI interpreter matching the declared Python environment")
        if not lock_ok: missing.append("installed packages matching the exact dependency lock (" + lock_reason + ")")
        selected = {(c["method"], c["backend"]) for c in args.case}
        if (("lp", "cvxpy") not in selected or not any(
            {(method, "numba"), (method, "rust")} <= selected for method in ("gda", "sgda"))):
            missing.append("CVXPY LP and a matching Numba/Rust learning pair")
        if args.repetitions < 5: missing.append("at least five repetitions")
        if args.repetitions % len(args.case):
            missing.append("a repetition count divisible by the number of cases for balanced order")
        if not hardware["cpu"] or hardware["cpu"].lower() in {"arm", "unknown"} or not hardware["memory_bytes"]:
            missing.append("CPU model or physical memory identity")
        if missing: parser.error("definitive run requires: " + ", ".join(missing))
    frozen_config = root / "config.toml"
    config_bytes = config.read_bytes()
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env.setdefault(key, "1")
    provenance = {"config_original": str(config), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                  "benchmark_script_sha256": digest(Path(__file__)), "command": command, "cwd": str(cwd),
                  "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                  "machine": platform.machine(), "processor": platform.processor(), "cases": args.case,
                  "source": source_identity(cwd), "environment": environment, "hardware": hardware,
                  "wheel": wheel, "rust_binary": rust, "cli_executable": cli_executable,
                  "environment_python": env_python, "wheel_binary_matches": matching_wheel, "dependency_lock": lock,
                  "wheel_python_matches": matching_python,
                  "lock_matches_environment": lock_ok, "lock_comparison": lock_reason,
                  "cli_environment": cli_env, "cli_matches_environment": same_environment,
                  "mode": args.mode, "numba_cache_state": args.numba_cache_state,
                  "numba_cache_dir": str(cache) if cache else None,
                  "prewarmed_cache_sha256": cache_identity(cache) if args.numba_cache_state == "prewarmed" else None,
                  "repetitions": args.repetitions, "seed": args.seed, "timeout": args.timeout, "rust_timeout": args.rust_timeout,
                  "allow_infeasible": args.allow_infeasible,
                  "recover_learning": args.recover,
                  "recovery_atol": args.recovery_atol, "recovery_rtol": args.recovery_rtol,
                  "atol": args.atol, "rtol": args.rtol, "bootstrap": args.bootstrap,
                  "output_dir": str(root), "workers": 1,
                  "runtime_environment": {k: env.get(k) for k in ("PATH", "PYTHONPATH", "MPLCONFIGDIR", "NUMBA_DISABLE_JIT")},
                  "thread_environment": {k: env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}}
    if root == cwd or cwd in root.parents:
        parser.error("output directory must be outside the source working directory")
    provenance["run_id"] = document_hash(provenance)
    run_id = provenance["run_id"]
    jobs = schedule(args.case, args.repetitions, args.seed)
    if args.resume:
        if not root.is_dir():
            parser.error("--resume requires an existing run directory")
    else:
        root.mkdir(parents=True, exist_ok=False)
    with run_lock(root):
        if args.resume:
            old = json.loads((root / "provenance.json").read_text())
            if old.get("run_id") != run_id or document_hash({k: v for k, v in old.items() if k != "run_id"}) != run_id:
                raise ValueError("run identity changed; start a separately identified run")
            if frozen_config.read_bytes() != config_bytes:
                raise ValueError("stored run configuration changed")
        else:
            (root / "shards").mkdir()
            (root / "attempts").mkdir()
            atomic_bytes(frozen_config, config_bytes)
            atomic_json(root / "provenance.json", provenance)
        committed = committed_shards(root, run_id, jobs)
        used_attempts = {payload["attempt"] for payload in committed.values()}
        orphans = sorted(p.name for p in (root / "attempts").iterdir() if p.name not in used_attempts)
        for name in orphans:
            orphan = root / "attempts" / name
            if not orphan.is_dir() or orphan.is_symlink() or orphan_worker_alive(orphan):
                raise ValueError(f"orphan attempt may still be active: {orphan}")
        prior_elapsed = 0.0
        if args.resume and (root / "progress.json").exists():
            prior_elapsed = float(json.loads((root / "progress.json").read_text()).get("elapsed_seconds", 0.0))
        report = progress_writer(root, len(jobs), prior_elapsed, orphans)
        terminal_path = root / "terminal.json"
        if terminal_path.exists():
            terminal = json.loads(terminal_path.read_text())
            ordered = [committed[f"{case['name']}-{rep:05}"]["record"] for case, rep in jobs] if len(committed) == len(jobs) else []
            if (terminal.get("run_id") != run_id or len(committed) != len(jobs)
                    or terminal.get("summary_sha256") != digest(root / "summary.json")
                    or terminal.get("receipts_sha256") != digest(root / "receipts.jsonl")
                    or terminal.get("provenance_sha256") != digest(root / "provenance.json")
                    or [json.loads(line) for line in (root / "receipts.jsonl").read_text().splitlines()] != ordered):
                raise ValueError("terminal artifacts failed verification")
            report(terminal["status"], "complete" if terminal["status"] == "complete" else "failed", len(committed))
            print(f"Verified terminal: {terminal_path}")
            return 0 if terminal.get("status") == "complete" else 1
        new_jobs = 0
        execution_segment = uuid.uuid4().hex
        for case, rep in jobs:
            job_id = f"{case['name']}-{rep:05}"
            if job_id in committed:
                continue
            attempt = f"{job_id}-{uuid.uuid4().hex}"
            folder = root / "attempts" / attempt
            folder.mkdir()
            job_env = env.copy()
            if args.numba_cache_state == "per-process-empty":
                job_cache = root / "numba-cache" / attempt
                job_cache.mkdir(parents=True)
            else:
                job_cache = cache
            job_env["NUMBA_CACHE_DIR"] = str(job_cache)
            report("running", "execution", len(committed), job_id)
            done = threading.Event()
            def heartbeat():
                while not done.wait(5):
                    report("running", "execution", len(committed), job_id)
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            worker_pid = []
            def worker_started(pid):
                atomic_json(folder / ".worker.json", {"status": "started", "pid": pid})
                worker_pid.append(pid)
            try:
                record = run_one(command, cwd, frozen_config, case, rep, folder, args.timeout, job_env,
                                 worker_started, args.rust_timeout, args.allow_infeasible, args.recover)
            except UncertainWorkerTreeError as exc:
                # Leave .worker.json in "started" state: on Windows resumption
                # fails closed while descendant liveness cannot be determined.
                report("interrupted", "execution", len(committed), job_id)
                print(f"benchmark: {exc}", file=sys.stderr)
                return 1
            finally:
                done.set()
                thread.join()
            if worker_pid:
                atomic_json(folder / ".worker.json", {"status": "finished", "pid": worker_pid[0]})
            record["numba_cache_dir"] = str(job_cache)
            record["execution_segment"] = execution_segment
            artifacts = {p.name: digest(p) for p in folder.iterdir() if p.is_file() and not p.is_symlink()}
            payload = {"run_id": run_id, "job_id": job_id, "attempt": attempt,
                       "row_count": 1, "record": record, "artifacts": artifacts}
            atomic_json(root / "shards" / f"{job_id}.json", {"payload": payload, "sha256": document_hash(payload)})
            committed[job_id] = payload
            new_jobs += 1
            report("running", "execution", len(committed))
            print(f"{rep + 1}/{args.repetitions} {case['name']}: {'ok' if record.get('artifact_verified') else record['error'] or record['exit_code']}", flush=True)
            if args.stop_after is not None and new_jobs >= args.stop_after and len(committed) < len(jobs):
                report("interrupted", "execution", len(committed))
                print("Stopped at a committed job boundary; resume with --resume.")
                return 130
        report("running", "collection", len(committed))
        committed = committed_shards(root, run_id, jobs)
        if len(committed) != len(jobs):
            raise ValueError("missing committed shards")
        records = [committed[f"{case['name']}-{rep:05}"]["record"] for case, rep in jobs]
        receipts = b"".join((json.dumps(row, allow_nan=False) + "\n").encode() for row in records)
        atomic_bytes(root / "receipts.jsonl", receipts)
        report("running", "aggregation", len(committed))
        summary = summarize(args.case, records, args.atol, args.rtol, args.bootstrap, args.seed,
                            args.recovery_atol, args.recovery_rtol)
        post_wheel = file_identity(wheel["path"]) if wheel else None
        post_rust = file_identity(rust["path"]) if rust else None
        post_environment = environment_identity(args.environment_python, cwd)
        post_source = source_identity(cwd)
        stable = ((not wheel or wheel == post_wheel) and (not rust or rust == post_rust)
                  and (not cli_executable or cli_executable == file_identity(cli_executable["path"]))
                  and (not env_python or env_python == file_identity(env_python["path"]))
                  and (not lock or lock == file_identity(lock["path"]))
                  and environment == post_environment and cli_env == cli_environment(command, cwd)
                  and (not matching_python or wheel_python_matches(args.environment_python, wheel["path"], cwd))
                  and (args.numba_cache_state != "prewarmed" or provenance["prewarmed_cache_sha256"] == cache_identity(cache))
                  and provenance["benchmark_script_sha256"] == digest(Path(__file__))
                  and (file_identity(str(frozen_config)) or {}).get("sha256") == provenance["config_sha256"]
                  and provenance["source"]["source_tree_sha256"] == post_source["source_tree_sha256"])
        summary["mode"] = args.mode
        summary["inputs_stable"] = stable
        summary["execution_segments"] = sorted({row["execution_segment"] for row in records})
        all_succeeded = all(r.get("artifact_verified") for r in records)
        all_matching = all(pair["all_pairs_match"] for pair in summary["paired"])
        if args.mode != "definitive" or not stable or not all_succeeded or not all_matching or len(summary["execution_segments"]) > 1:
            for pair in summary["paired"]:
                pair["ratio_estimate"] = None
                for item in pair["pairs"]:
                    item["wall_ratio_baseline_over_candidate"] = None
            summary["ratio_suppressed_reason"] = ("qualification run" if args.mode != "definitive" else
                "run used multiple execution segments" if len(summary["execution_segments"]) > 1 else
                "execution inputs changed or a run/parity check failed")
        atomic_json(root / "summary.json", summary)
        report("running", "terminal_verification", len(committed))
        # Reload all aggregate inputs and outputs before publishing a terminal state.
        reloaded = committed_shards(root, run_id, jobs)
        if reloaded != committed or json.loads((root / "summary.json").read_text()) != summary:
            raise ValueError("terminal aggregate reload failed")
        if [json.loads(line) for line in (root / "receipts.jsonl").read_text().splitlines()] != records:
            raise ValueError("terminal receipts reload failed")
        success = all_succeeded and all_matching and stable
        terminal = {"run_id": run_id, "status": "complete" if success else "failed",
                    "completed": len(committed), "total": len(jobs),
                    "summary_sha256": digest(root / "summary.json"),
                    "receipts_sha256": digest(root / "receipts.jsonl"),
                    "provenance_sha256": digest(root / "provenance.json")}
        atomic_json(terminal_path, terminal)
        if json.loads(terminal_path.read_text()) != terminal:
            raise ValueError("terminal reload failed")
        report(terminal["status"], "complete" if success else "failed", len(committed))
        print(f"Summary: {root / 'summary.json'}")
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
