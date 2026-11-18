"""Python interface for the lplearn Rust backend and legacy API."""

from .backend import BackendError, backend_info, backend_ping

_LEGACY_EXPORTS = (
    "calculate_cost", "instantaneous_cost", "processing_cost", "storage_cost",
    "compute_lagrangian", "solver", "learner", "Model", "compute_U", "compute_V",
    "estimate_P", "compute_P", "compute_Q", "inst_prob", "inst_MDP",
    "simulate_next_state", "acceptance", "highness", "fullness", "kappa",
    "get_updated_params", "verify_aggregation", "compare_results", "METHODS",
    "PARAMETERS_DESCRIPTION", "policy_pie_plot",
)

__all__ = [
    "BackendError", "backend_info", "backend_ping", "load_config", "solve",
    "recover_occupation", *_LEGACY_EXPORTS,
]


def __getattr__(name):
    if name in ("load_config", "solve", "recover_occupation"):
        from . import workflow
        return getattr(workflow, name)
    if name in _LEGACY_EXPORTS:
        from importlib import import_module
        legacy = import_module("._legacy", __name__)
        return getattr(legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
