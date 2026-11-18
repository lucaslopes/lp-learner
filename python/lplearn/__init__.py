"""Python interface for the lplearn Rust backend."""

from .backend import BackendError, backend_info, backend_ping

__all__ = ["BackendError", "backend_info", "backend_ping", "load_config", "solve"]


def __getattr__(name):
    if name in ("load_config", "solve"):
        from . import workflow
        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
