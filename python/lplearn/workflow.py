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


def solve(config, method, backend):
    """Return occupation measures y[t,s,u] and metadata for a single job."""
    if (method, backend) not in {("lp", "cvxpy"), ("gda", "numba"), ("sgda", "numba")}:
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
        result = run_gda(model, **settings)
        # Every returned iterate, trace and diagnostic belongs to this run.
        # Reject divergence even if the selected average happens to be finite.
        for name, value in result.items():
            if isinstance(value, (np.ndarray, float, np.floating)) and not np.isfinite(value).all():
                raise RuntimeError(f"Numba solver returned non-finite values in {name}")
        y = result["y_avg"]
        metadata = {"seed": result["seed"], "iterations": result["n_iter"],
                    "samples_per_pair": result["samples_per_pair"], "averaged_iterations": result["n_avg"],
                    "solver_reported_seconds": result["wall_time"],
                    "solver_timing_scope": "Numba _run including any JIT compilation"}
    return np.asarray(y, dtype=float), {"method": method, "backend": backend,
                                      "model_seconds": model_seconds,
                                      "solve_seconds": perf_counter() - start, **metadata}
