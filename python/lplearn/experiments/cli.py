'''
Command line for the experiment runner: ``lplearn experiments``.

Exit codes: 0 success; 1 a job failed, or status found incomplete jobs; 2 bad input or stage error;
130 interrupted (completed jobs are kept, and the next run resumes).
'''

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from lplearn.experiments import runner
from lplearn.experiments.runner import ExperimentError, RunReport

DEFAULT_CONFIG_DIR = Path('configs')


def build_parser() -> argparse.ArgumentParser:
    '''
    Argument parser with one sub-command per action.
    '''

    parser = argparse.ArgumentParser(
        prog='lplearn experiments',
        description='Resumable, parallel runner for the numerical experiments.',
    )
    commands = parser.add_subparsers(dest='command', required=True, metavar='COMMAND')

    commands.add_parser('list', help='list the registered stages').set_defaults(handler=_cmd_list)

    run = commands.add_parser('run', help='run the jobs of a stage that are not complete')
    _add_stage_arguments(run)
    run.add_argument('--workers', type=int, default=None, help=f'worker processes (default: {runner.default_workers()})')
    run.add_argument('--force', action='store_true', help='re-run jobs that are already complete')
    run.add_argument('--dry-run', action='store_true', help='print the plan and run nothing')
    run.add_argument(
        '--min-free-gb',
        type=float,
        default=runner.DEFAULT_MIN_FREE_GB,
        help='hold new jobs while less memory than this is available, in GB (default: %(default)s)',
    )
    run.add_argument(
        '--only',
        action='append',
        metavar='JOB_ID_PREFIX',
        help='run only jobs whose id starts with this prefix (repeatable)',
    )
    run.set_defaults(handler=_cmd_run)

    summarize = commands.add_parser('summarize', help='write the stage tables from its complete results')
    _add_stage_arguments(summarize)
    summarize.add_argument('--out-dir', default=None, help='output directory (default: {results}/{stage}/summary)')
    summarize.set_defaults(handler=_cmd_summarize)

    status = commands.add_parser('status', help='count complete, failed, incomplete and missing jobs')
    _add_stage_arguments(status)
    status.set_defaults(handler=_cmd_status)
    return parser


def _add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('stage', help='stage name, as shown by `list`')
    parser.add_argument('--config', default=None, help='TOML config (default: configs/<stage>.toml)')
    parser.add_argument('--results-dir', default=None, help='results root (default: $LPL_RESULTS_DIR, else ./results)')
    parser.add_argument('--smoke', action='store_true', help='apply smoke overrides and write under <results>/smoke')


def _config_path(args: argparse.Namespace) -> Path:
    return Path(args.config) if args.config else DEFAULT_CONFIG_DIR / f'{args.stage}.toml'


def _cmd_list(_: argparse.Namespace) -> int:
    for name, module_path, found in runner.stage_listing():
        print(f'{name:<14} {module_path:<46} {"available" if found else "not written yet"}')
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    report = runner.run_stage(
        args.stage,
        _config_path(args),
        results_root=args.results_dir,
        workers=args.workers,
        force=args.force,
        dry_run=args.dry_run,
        smoke=args.smoke,
        min_free_gb=args.min_free_gb,
        only=args.only,
    )
    if report.status == 'dry-run':
        _print_plan(report)
        return 0
    return 0 if report.ok else 1


def _print_plan(report: RunReport) -> None:
    print(f'# {report.stage}: {report.total} jobs selected, {report.total - report.skipped} to run, {report.skipped} complete')
    for item in report.plan:
        print(f'{item.job_id}  {item.action:<4}  {item.reason:<10}  {json.dumps(item.job, sort_keys=True)}')


def _cmd_summarize(args: argparse.Namespace) -> int:
    out = runner.summarize_stage(
        args.stage,
        _config_path(args),
        results_root=args.results_dir,
        out_dir=args.out_dir,
        smoke=args.smoke,
    )
    print(out)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    report = runner.status_stage(args.stage, _config_path(args), results_root=args.results_dir, smoke=args.smoke)
    print(f'{report.stage} v{report.version}: {report.total} jobs in {report.results_dir}')
    for key in ('complete', 'failed', 'incomplete', 'missing'):
        print(f'  {key:<11}{report.counts.get(key, 0)}')
    for job_id, error in report.failed:
        print(f'  failed {job_id}: {error}')
    return 0 if report.complete else 1


def main(argv: Sequence[str] | None = None) -> int:
    '''
    Entry point. Returns the process exit code (see the module docstring).
    '''

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stderr)
    try:
        return args.handler(args)
    except ExperimentError as exc:
        print(f'lplearn experiments: error: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('lplearn experiments: interrupted; completed jobs are kept and the next run resumes', file=sys.stderr)
        return 130
