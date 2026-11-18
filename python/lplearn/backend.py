"""The versioned, process-based Rust backend health interface."""

import json
from importlib import metadata
import subprocess
import sys

SCHEMA = "lplearn-bridge/1"


class BackendError(RuntimeError):
    """The packaged backend could not complete a request."""


def _request(command: str, *, timeout: float = 30.0) -> dict:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    suffix = ".exe" if sys.platform == "win32" else ""
    try:
        dist = metadata.distribution("lplearn")
        matches = [dist.locate_file(item) for item in dist.files or ()
                   if item.name == f"lplearn-native{suffix}"]
    except metadata.PackageNotFoundError as error:
        raise BackendError("lplearn distribution metadata is unavailable") from error
    if len(matches) != 1:
        raise BackendError("packaged Rust backend not found in installed distribution")
    executable = matches[0]
    try:
        completed = subprocess.run(
            [str(executable)],
            input=json.dumps({"schema": SCHEMA, "command": command}),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise BackendError(f"Rust backend timed out after {timeout:g}s") from error
    except OSError as error:
        raise BackendError(f"Cannot execute Rust backend at {executable}: {error}") from error
    if completed.returncode:
        raise BackendError(f"Rust backend exited with {completed.returncode}: {completed.stderr.strip()}")
    try:
        response = json.loads(completed.stdout)
    except (ValueError, TypeError) as error:
        raise BackendError("Rust backend returned invalid JSON") from error
    if (not isinstance(response, dict) or response.get("schema") != SCHEMA
            or response.get("command") != command or response.get("backend") != "rust"
            or not isinstance(response.get("version"), str) or response.get("capabilities") != []):
        raise BackendError("Rust backend returned an incompatible response")
    return response


def backend_ping(*, timeout: float = 30.0) -> dict:
    return _request("ping", timeout=timeout)


def backend_info(*, timeout: float = 30.0) -> dict:
    return _request("info", timeout=timeout)
