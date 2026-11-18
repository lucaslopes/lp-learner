"""Rust implementation of the reference run_gda contract via lpl-core/1."""

from importlib import metadata
import json
import math
import subprocess
import sys
import time

import numpy as np


class RustBackendError(RuntimeError):
    """The native solver failed or returned an incompatible response."""


def _executable():
    name = "lplearn-solver.exe" if sys.platform == "win32" else "lplearn-solver"
    try:
        dist = metadata.distribution("lplearn")
        matches = [dist.locate_file(item) for item in dist.files or () if item.name == name]
    except metadata.PackageNotFoundError as error:
        raise RustBackendError("lplearn distribution metadata is unavailable") from error
    if len(matches) != 1:
        raise RustBackendError("packaged Rust solver not found in installed distribution")
    return matches[0]


def run_gda(model, Gamma_y, Gamma_lambda, Gamma_mu, eta_y, eta_dual, n_iter,
            samples_per_pair=0, seed=0, avg_window=None, record_every=None,
            init="uniform", state_weights=None, *, timeout=30.0, executable=None):
    """Run GDA or SGDA in Rust, preserving the Python reference result layout."""
    if init not in ("uniform", "zeros"):
        raise ValueError("init must be 'uniform' or 'zeros'")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    shapes = ((model.T, model.S, model.U), (model.L, model.T),
              (model.T, model.S, model.U), (model.S,), (model.T - 1, model.S))
    rng = np.random.default_rng(seed)
    state = np.concatenate([(rng.random(shape) if init == "uniform" else np.zeros(shape)).ravel()
                            for shape in shapes])
    weights = np.ones(model.S) if state_weights is None else np.asarray(state_weights, dtype=np.float64)
    if weights.shape != (model.S,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("state_weights must be a positive finite vector with one entry per state")
    request = {
        "schema": "lpl-core/1", "command": "solve",
        "model": {"states": model.S, "actions": model.U, "resources": model.L,
                  "horizon": model.T, "p": model.P.ravel().tolist(),
                  "c": model.c.ravel().tolist(), "d": model.d.ravel().tolist(),
                  "budgets": model.budgets.tolist(), "initial": model.m0.tolist()},
        "initial_state": state.tolist(),
        "parameters": {"gamma_y": float(Gamma_y), "gamma_lambda": float(Gamma_lambda),
                       "gamma_mu": float(Gamma_mu), "eta_y": float(eta_y),
                       "eta_dual": float(eta_dual), "state_weights": weights.tolist()},
        "iterations": int(n_iter), "samples_per_pair": int(samples_per_pair),
        "seed": int(seed) % (2**32),
        "record_every": max(1, n_iter // 200) if record_every is None else int(record_every),
    }
    if avg_window is not None:
        request["average_window"] = int(avg_window)
    start = time.perf_counter()
    try:
        response = subprocess.run([str(executable if executable is not None else _executable())],
                                  input=json.dumps(request, allow_nan=False) + "\n", text=True,
                                  capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise RustBackendError(f"Rust solver timed out after {timeout:g}s") from error
    except OSError as error:
        raise RustBackendError(f"cannot execute Rust solver: {error}") from error
    if response.returncode:
        raise RustBackendError(f"Rust solver exited with {response.returncode}: {response.stderr.strip()}")
    elapsed = time.perf_counter() - start
    try:
        wire = json.loads(response.stdout)
        if wire["schema"] != "lpl-core/1":
            raise ValueError("incorrect schema")
        execute_seconds = float(wire["execute_seconds"])
        backend_version = wire["backend_version"]
        debug_assertions = wire["debug_assertions"]
        if (not math.isfinite(execute_seconds) or execute_seconds < 0
                or not isinstance(backend_version, str) or not backend_version
                or not isinstance(debug_assertions, bool)):
            raise ValueError("incorrect execution metadata")
        out = wire["output"]
        result = {}
        for which, key in (("last", "last"), ("avg", "average")):
            flat = np.asarray(out[key], dtype=float)
            if flat.ndim != 1 or flat.size != sum(int(np.prod(shape)) for shape in shapes):
                raise ValueError("incorrect iterate length")
            offset = 0
            for name, shape in zip(("y", "lam_res", "lam_nn", "mu0", "muf"), shapes):
                size = int(np.prod(shape))
                result[f"{name}_{which}"] = flat[offset:offset + size].reshape(shape)
                offset += size
        n_records = len(out["trace_iter"])
        result.update(n_avg=out["n_average"], trace_iter=np.asarray(out["trace_iter"], dtype=np.int64),
                      trace_y=np.asarray(out["trace_y"], dtype=float).reshape(n_records, *shapes[0]),
                      trace_lam_res=np.asarray(out["trace_resource_dual"], dtype=float).reshape(n_records, *shapes[1]),
                      trace_mu_norm=np.asarray(out["trace_mu_norm"], dtype=float))
        if result["trace_mu_norm"].shape != (n_records,):
            raise ValueError("incorrect trace length")
        if (not isinstance(result["n_avg"], int) or isinstance(result["n_avg"], bool)
                or not 0 <= result["n_avg"] <= int(n_iter)):
            raise ValueError("incorrect average count")
        if any(not np.isfinite(value).all() for value in result.values()
               if isinstance(value, np.ndarray)):
            raise ValueError("nonfinite solver output")
    except (KeyError, TypeError, ValueError) as error:
        raise RustBackendError("Rust solver returned an incompatible response") from error
    result.update(wall_time=elapsed, execute_seconds=execute_seconds,
                  backend_version=backend_version, debug_assertions=debug_assertions,
                  seed=int(seed), n_iter=int(n_iter),
                  samples_per_pair=int(samples_per_pair), eta_y=float(eta_y),
                  eta_dual=float(eta_dual), state_weights=weights,
                  implementation="rust-process-json")
    return result
