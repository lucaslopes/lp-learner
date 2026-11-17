"""Command-line interface for the installed lplearn package."""

import argparse
import json
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
    args = parser.parse_args(argv)
    try:
        result = (backend_ping if args.action == "ping" else backend_info)(timeout=args.timeout)
    except (BackendError, ValueError) as error:
        print(f"lplearn: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0
