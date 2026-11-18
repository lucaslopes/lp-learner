"""Single-configuration queue model, LP, and learning workflow."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import math
import tomllib

import numpy as np

from .core import QueueParams, build_lp, queue_wcmdp, run_gda, solve_lp


def load_config(path):
    """Load a single model and learner from TOML, filling in defaults."""
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    unknown = set(config) - {"model", "learning"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    if not isinstance(config.get("model", {}), dict) or not isinstance(config.get("learning", {}), dict):
        raise ValueError("model and learning must be TOML tables")
    model = dict(config.get("model", {}))
    if model.get("m0") is not None:
        model["m0"] = tuple(model["m0"])
    params = QueueParams(**model)
    if any(not isinstance(getattr(params, k), int) or isinstance(getattr(params, k), bool)
           for k in ("K", "T")):
        raise ValueError("K and T must be integers")
    numeric = (params.p, params.b_low, params.b_high, params.alpha, params.beta,
               params.gamma, params.gamma_bar)
    if not all(math.isfinite(v) for v in numeric):
        raise ValueError("model parameters must be finite")
    if not 0 <= params.alpha <= 1 or not 0 <= params.beta <= 1:
        raise ValueError("alpha and beta must be between 0 and 1")
    if params.m0 is not None and (any(not math.isfinite(v) or v < 0 for v in params.m0)
                                  or not math.isfinite(sum(params.m0)) or sum(params.m0) <= 0):
        raise ValueError("m0 must contain finite nonnegative values with positive sum")
    defaults = dict(Gamma_y=1.0, Gamma_lambda=2e-5, Gamma_mu=2e-5,
                    eta_y=0.9, eta_dual=0.03, n_iter=1000,
                    samples_per_pair=10, seed=0, avg_window=100,
                    record_every=100, init="uniform")
    learning = dict(config.get("learning", {}))
    unknown = set(learning) - set(defaults)
    if unknown:
        raise ValueError(f"unknown learning settings: {sorted(unknown)}")
    defaults.update(learning)
    for key in ("n_iter", "samples_per_pair", "avg_window", "record_every", "seed"):
        value = defaults[key]
        nonnegative = key in ("seed", "samples_per_pair")
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if nonnegative else 1):
            raise ValueError(f"{key} must be a {'nonnegative' if nonnegative else 'positive'} integer")
    if defaults["init"] not in ("uniform", "zeros"):
        raise ValueError("init must be 'uniform' or 'zeros'")
    for key in ("Gamma_y", "Gamma_lambda", "Gamma_mu", "eta_y", "eta_dual"):
        value = defaults[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if value < 0 or (key.startswith("eta_") and value == 0):
            raise ValueError(f"invalid {key}")
    return {"model": asdict(params), "learning": defaults}


def occupation_feasibility(occupation, model, *, tolerance=1e-6):
    """Report maximum LP constraint violations on the original occupation array.

    Feasibility is assessed against the representative-arm relaxation, not
    sample-path constraints for a finite population.
    """
    y = np.asarray(occupation, dtype=float)
    if y.shape != (model.T, model.S, model.U) or not np.isfinite(y).all():
        raise ValueError("occupation has an invalid shape or nonfinite values")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("feasibility tolerance must be finite and nonnegative")
    mass = y.sum(axis=2)
    initial = np.max(np.abs(mass[0] - model.m0))
    if model.T > 1:
        inflow = np.einsum("tsu,usk->tk", y[:-1], model.P)
        flow = np.max(np.abs(mass[1:] - inflow))
    else:
        flow = 0.0
    resource = np.einsum("tsu,lsu->lt", y, model.d) - model.budgets[:, None]
    values = {
        "min_occupation": float(y.min()),
        "nonnegativity_violation": float(max(0.0, -y.min())),
        "initial_residual": float(initial),
        "flow_residual": float(flow),
        "stage_mass_residual": float(np.max(np.abs(mass.sum(axis=1) - 1.0))),
        "resource_violation": float(max(0.0, np.max(resource))),
    }
    values["tolerance"] = float(tolerance)
    values["feasible"] = all(values[k] <= tolerance for k in (
        "nonnegativity_violation", "initial_residual", "flow_residual",
        "stage_mass_residual", "resource_violation"))
    values["scope"] = "representative-arm LP relaxation"
    return values


def recover_occupation(occupation, model, *, tolerance=1e-6):
    """Project an iterate onto the representative-arm LP feasible set.

    This changes the learning result. It does not enforce sample-path budgets
    for a finite population or establish optimizer convergence.
    """
    import cvxpy as cp

    y = np.asarray(occupation, dtype=float)
    raw = occupation_feasibility(y, model, tolerance=tolerance)
    struct = build_lp(model)
    x = cp.Variable(struct.n_y)
    objective = cp.Minimize(cp.sum_squares(x - y.ravel()))
    problem = cp.Problem(objective, [struct.A @ x == struct.b, struct.D @ x <= struct.r])
    start = perf_counter()
    try:
        problem.solve(solver="CLARABEL")
    except cp.error.SolverError as error:
        raise RuntimeError(f"occupation recovery failed: {error}") from error
    elapsed = perf_counter() - start
    if (problem.status not in ("optimal", "optimal_inaccurate") or x.value is None
            or problem.value is None or not np.isfinite(problem.value)):
        raise RuntimeError(f"occupation recovery failed: {problem.status}")
    recovered = np.asarray(x.value, dtype=float).reshape(y.shape)
    checked = occupation_feasibility(recovered, model, tolerance=tolerance)
    if not checked["feasible"]:
        raise RuntimeError(f"occupation recovery did not meet tolerance: {checked}")
    return recovered, {
        "method": "euclidean_projection_lp_relaxation",
        "solver": "CLARABEL", "status": problem.status,
        "seconds": elapsed, "distance_l2": float(np.linalg.norm(recovered - y)),
        "objective_value": float(problem.value),
        "original_feasibility": raw, "recovered_feasibility": checked,
    }


def solve(config, method, backend, *, rust_timeout=30.0):
    """Return occupation measures y[t,s,u] and metadata for a single job."""
    if (method, backend) not in {("lp", "cvxpy"), ("gda", "numba"), ("sgda", "numba"),
                                  ("gda", "rust"), ("sgda", "rust")}:
        raise ValueError(f"unsupported method/backend combination: {method}/{backend}")
    if method == "sgda" and config["learning"]["samples_per_pair"] < 1:
        raise ValueError("SGDA requires a positive samples_per_pair")
    fields = dict(config["model"])
    if fields.get("m0") is not None:
        fields["m0"] = tuple(fields["m0"])
    start = perf_counter()
    model = queue_wcmdp(QueueParams(**fields))
    model_seconds = perf_counter() - start
    start = perf_counter()
    if method == "lp":
        result = solve_lp(build_lp(model))
        y = result["y"]
        if not np.isfinite(y).all() or not np.isfinite(result["objective"]):
            raise RuntimeError("CVXPY solver returned non-finite results")
        reported = result["solve_time"]
        metadata = {"solver": result["solver"], "status": result["status"],
                    "objective": result["objective"],
                    "solver_reported_seconds": reported if np.isfinite(reported) else None}
    else:
        settings = dict(config["learning"])
        if method == "gda":
            settings["samples_per_pair"] = 0
        if backend == "rust":
            from .rust import run_gda as native_run_gda
            result = native_run_gda(model, **settings, timeout=rust_timeout)
        else:
            result = run_gda(model, **settings)
        # Reject divergence in any returned iterate, trace or diagnostic.
        for name, value in result.items():
            if isinstance(value, (np.ndarray, float, np.floating)) and not np.isfinite(value).all():
                raise RuntimeError(f"{backend} solver returned non-finite values in {name}")
        y = result["y_avg"]
        metadata = {"seed": result["seed"], "iterations": result["n_iter"],
                    "samples_per_pair": result["samples_per_pair"], "averaged_iterations": result["n_avg"]}
        if backend == "rust":
            metadata.update(solver_reported_seconds=result["execute_seconds"],
                            solver_timing_scope="Rust execution including model and solver setup",
                            backend_process_seconds=result["wall_time"],
                            backend_version=result["backend_version"],
                            backend_debug_assertions=result["debug_assertions"])
        else:
            metadata.update(solver_reported_seconds=result["wall_time"],
                            solver_timing_scope="Numba _run including any JIT compilation")
    solve_seconds = perf_counter() - start
    start = perf_counter()
    metadata["occupation_feasibility"] = occupation_feasibility(y, model)
    metadata["feasibility_seconds"] = perf_counter() - start
    return np.asarray(y, dtype=float), {"method": method, "backend": backend,
                                      "model_seconds": model_seconds,
                                      "solve_seconds": solve_seconds, **metadata}
