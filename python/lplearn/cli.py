"""Command-line interface for the installed lplearn package."""

import argparse
import json
from pathlib import Path
import sys

from .backend import BackendError, backend_info, backend_ping


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lplearn")
    commands = parser.add_subparsers(dest="section", required=True)
    backend = commands.add_parser("backend", help="inspect the Rust backend")
    actions = backend.add_subparsers(dest="action", required=True)
    for action in ("ping", "info"):
        sub = actions.add_parser(action)
        sub.add_argument("--timeout", type=float, default=30.0, help="worker timeout in seconds")
    solve = commands.add_parser("solve", help="solve a single configured queue model")
    solve.add_argument("--config", required=True, help="TOML configuration file")
    solve.add_argument("--method", required=True, choices=("lp", "gda", "sgda"))
    solve.add_argument("--backend", required=True, choices=("cvxpy", "numba", "rust"))
    solve.add_argument("--rust-timeout", type=float, default=30.0,
                       help="Rust solver timeout in seconds (default: 30)")
    solve.add_argument("--output", required=True, help="output .npz file")
    plot = commands.add_parser("plot-policy", help="solve and plot occupation measures")
    plot.add_argument("--config", required=True, help="single-job TOML configuration")
    plot.add_argument("--method", required=True, choices=("lp", "gda", "sgda"))
    plot.add_argument("--backend", required=True, choices=("cvxpy", "numba", "rust"))
    plot.add_argument("--rust-timeout", type=float, default=30.0, help="Rust solver timeout in seconds")
    plot.add_argument("--output", help="output plot file (default: results/<config>-<method>-<backend>.png)")
    plot.add_argument("--format", choices=("png", "pdf", "svg"), help="plot format")
    recovery = plot.add_mutually_exclusive_group()
    recovery.add_argument("--recover", action="store_true",
                          help="project onto the LP relaxation before plotting")
    recovery.add_argument("--allow-infeasible", action="store_true",
                      help="plot an infeasible iterate as a clearly marked diagnostic")
    plot.add_argument("--open", action="store_true", help="open the saved plot")
    args = parser.parse_args(argv)
    try:
        if args.section == "solve":
            return _solve(args)
        if args.section == "plot-policy":
            from .plotting import plot_command
            return plot_command(args)
        result = (backend_ping if args.action == "ping" else backend_info)(timeout=args.timeout)
    except (BackendError, ValueError, TypeError, OSError, RuntimeError) as error:
        print(f"lplearn: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


def _solve(args):
    import numpy as np
    from .workflow import load_config, solve

    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("output must end in .npz")
    config = load_config(args.config)
    occupation, metadata = solve(config, args.method, args.backend, rust_timeout=args.rust_timeout)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_json = json.dumps(config, allow_nan=False)
    metadata_json = json.dumps(metadata, allow_nan=False)
    with output.open("wb") as file:
        np.savez_compressed(file, occupation=occupation,
                            config_json=config_json, metadata_json=metadata_json)
    print(f"Occupation measures: {output}")
    return 0
