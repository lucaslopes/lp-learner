'''
Experiment runner: resumable, parallel execution of stage jobs.

A stage (see lplearn/experiments/stages) turns a config into a list of JSON-serializable
jobs and runs one job at a time. Each result is stored as ``{results}/{stage}/{job_id}.npz``
next to a ``{job_id}.json`` record. A job counts as complete when its record says ``ok`` and
its arrays load, so re-running a stage computes only what is missing or failed.
'''

import hashlib
import importlib
import importlib.util
import json
import logging
import math
import multiprocessing
import os
import re
try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None
import signal
import sys
import time
import tomllib
import traceback
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import psutil

from lplearn.experiments import provenance
from lplearn.experiments.io import RESERVED_NPZ_KEYS, atomic_savez, atomic_write_json, load_arrays, read_json
from lplearn.experiments.stages import REGISTRY

logger = logging.getLogger('lplearn.experiments')

RESULTS_ENV_VAR = 'LPL_RESULTS_DIR'
DEFAULT_RESULTS_DIR = 'results'
DEFAULT_MIN_FREE_GB = 4.0
MEMORY_POLL_S = 5.0
THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS',
)
RECORD_SCHEMA = 1
MANIFEST_SCHEMA = 2  # Separates caller-project provenance from installed runtime identity.
_RESERVED_STAGE_NAMES = frozenset({'manifests'})
_GIB = 2**30


class ExperimentError(Exception):
    '''
    Base class for errors that the command line reports without a traceback.
    '''


class StageError(ExperimentError):
    '''
    A stage is unknown, cannot be imported, or does not follow the stage contract.
    '''


class ConfigError(ExperimentError):
    '''
    A config file or a runner option is invalid.
    '''


class JobState(str, Enum):
    '''
    Where a job stands on disk.
    '''

    COMPLETE = 'complete'
    FAILED = 'failed'
    INCOMPLETE = 'incomplete'
    MISSING = 'missing'


@dataclass(frozen=True)
class LoadedConfig:
    '''
    A parsed config, with the SHA-256 of the file it came from.
    '''

    raw: dict[str, Any]
    path: Path | None
    sha256: str


@dataclass(frozen=True)
class JobEntry:
    '''
    A job from make_jobs, with its identifier.
    '''

    job_id: str
    job: dict[str, Any]


@dataclass(frozen=True)
class PlannedJob:
    '''
    What a run does with one job: action ``run`` or ``skip``, and the reason.
    '''

    job_id: str
    action: str
    reason: str
    job: dict[str, Any]


@dataclass(frozen=True)
class JobOutcome:
    '''
    The result of executing one job, sent from the worker back to the parent.
    '''

    job_id: str
    status: str
    wall_time_s: float
    worker_pid: int | None
    peak_rss_mb: float | None
    error: str | None


@dataclass(frozen=True)
class RunReport:
    '''
    Summary of a run_stage call. ``status`` is ``dry-run``, ``ok`` or ``failed``.
    '''

    stage: str
    results_dir: Path
    plan: tuple[PlannedJob, ...]
    total: int
    skipped: int
    executed: int
    failed: tuple[str, ...]
    status: str
    manifest_path: Path | None = None
    log_path: Path | None = None

    @property
    def ok(self) -> bool:
        '''
        True when no job failed in this run.
        '''

        return not self.failed


@dataclass(frozen=True)
class StatusReport:
    '''
    Counts of complete, failed, incomplete and missing jobs for a config.
    '''

    stage: str
    version: str
    results_dir: Path
    total: int
    counts: dict[str, int]
    failed: tuple[tuple[str, str], ...]

    @property
    def complete(self) -> bool:
        '''
        True when every job of the config is complete.
        '''

        return self.counts[JobState.COMPLETE.value] == self.total


@dataclass(frozen=True)
class _Prepared:
    module: ModuleType
    loaded: LoadedConfig
    cfg: dict[str, Any]
    entries: list[JobEntry]


# ---------------------------------------------------------------------------
# Configs and stages
# ---------------------------------------------------------------------------


def load_config(source: str | os.PathLike | Mapping[str, Any]) -> LoadedConfig:
    '''
    Parse a TOML config file, or accept a mapping, and hash the bytes for provenance.
    '''

    if isinstance(source, Mapping):
        raw = dict(source)
        digest = provenance.sha256_bytes(json.dumps(raw, sort_keys=True, default=str).encode('utf-8'))
        return LoadedConfig(raw=raw, path=None, sha256=digest)

    path = Path(source)
    if not path.is_file():
        raise ConfigError(f'config file not found: {path}')
    data = path.read_bytes()
    try:
        raw = tomllib.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f'{path} is not valid TOML: {exc}') from exc
    return LoadedConfig(raw=raw, path=path.resolve(), sha256=provenance.sha256_bytes(data))


def merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    '''
    Recursively merge `override` into a copy of `base`: nested tables merge, other values are replaced.
    '''

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def effective_config(raw: Mapping[str, Any], stage_smoke: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    '''
    The config a stage receives. In smoke mode the stage's SMOKE overrides apply first and the config's
    own [smoke] table after them, so the most specific setting wins. The [smoke] table is never passed on.
    '''

    body = {key: value for key, value in raw.items() if key != 'smoke'}
    if not smoke:
        return body
    smoke_table = raw.get('smoke', {})
    if not isinstance(smoke_table, Mapping):
        raise ConfigError('the [smoke] entry of a config must be a table')
    return merge_config(merge_config(body, stage_smoke), smoke_table)


def load_stage(name: str) -> ModuleType:
    '''
    Import the stage registered as `name` (lazily) and check that it follows the stage contract.
    '''

    if name not in REGISTRY:
        raise StageError(f'unknown stage {name!r}; registered stages: {", ".join(sorted(REGISTRY))}')
    module_path = REGISTRY[name]
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name and (module_path == exc.name or module_path.startswith(exc.name + '.')):
            raise StageError(f'stage {name!r} is registered as {module_path!r}, but that module does not exist yet') from exc
        raise StageError(f'stage {name!r} could not be imported: no module named {exc.name!r}') from exc
    except Exception as exc:
        raise StageError(f'stage {name!r} failed to import: {type(exc).__name__}: {exc}') from exc
    validate_stage(module)
    if module.NAME != name:
        raise StageError(f'{module_path} declares NAME={module.NAME!r} but is registered as {name!r}')
    return module


def validate_stage(module: ModuleType) -> None:
    '''
    Raise StageError unless `module` defines NAME and VERSION (strings), make_jobs and run_job (functions),
    and SMOKE (a mapping), if present.
    '''

    label = module.__name__
    for attr in ('NAME', 'VERSION'):
        if not isinstance(getattr(module, attr, None), str):
            raise StageError(f'{label} must define {attr} as a string')
    if not re.fullmatch(r'[a-z][a-z0-9_]*', module.NAME) or module.NAME in _RESERVED_STAGE_NAMES:
        raise StageError(f'{label}: NAME {module.NAME!r} must be lowercase snake_case and not reserved')
    for attr in ('make_jobs', 'run_job'):
        if not callable(getattr(module, attr, None)):
            raise StageError(f'{label} must define a function {attr}()')
    if not isinstance(getattr(module, 'SMOKE', {}), Mapping):
        raise StageError(f'{label}: SMOKE must be a mapping of config overrides')


def stage_listing() -> list[tuple[str, str, bool]]:
    '''
    (name, module path, importable) for every registered stage, without importing the stages.
    '''

    rows = []
    for name, module_path in sorted(REGISTRY.items()):
        try:
            found = importlib.util.find_spec(module_path) is not None
        except (ImportError, ValueError):
            found = False
        rows.append((name, module_path, found))
    return rows


def _resolve_stage(stage: str | ModuleType) -> ModuleType:
    if isinstance(stage, str):
        return load_stage(stage)
    validate_stage(stage)
    if stage.__name__ == '__main__' or sys.modules.get(stage.__name__) is not stage:
        raise StageError(f'{stage.__name__}: a stage object must be an importable module, because worker processes import it by name')
    return stage


def _prepare(stage: str | ModuleType, config: str | os.PathLike | Mapping[str, Any], *, smoke: bool) -> _Prepared:
    module = _resolve_stage(stage)
    loaded = load_config(config)
    cfg = effective_config(loaded.raw, getattr(module, 'SMOKE', {}), smoke=smoke)
    return _Prepared(module=module, loaded=loaded, cfg=cfg, entries=build_job_table(module, cfg))


# ---------------------------------------------------------------------------
# Job identity and layout
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    '''
    Compact JSON with sorted keys, used for job identities. NaN, infinities and non-JSON types are refused.
    '''

    try:
        return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StageError(f'job parameters must be plain JSON values: {exc}') from exc


def job_id(stage: str, version: str, job: Mapping[str, Any]) -> str:
    '''
    16 hex digits: the SHA-256 of the canonical JSON of {stage, stage VERSION, job}.
    '''

    payload = canonical_json({'stage': stage, 'version': version, 'job': job})
    return hashlib.sha256(payload.encode('ascii')).hexdigest()[:16]


def build_job_table(module: ModuleType, cfg: Mapping[str, Any]) -> list[JobEntry]:
    '''
    Call make_jobs(cfg) and attach identifiers. Duplicate jobs are refused, since they would share one result.
    '''

    entries: list[JobEntry] = []
    seen: set[str] = set()
    for job in module.make_jobs(dict(cfg)):
        if not isinstance(job, dict):
            raise StageError(f'{module.NAME}.make_jobs must return dicts, got {type(job).__name__}')
        identifier = job_id(module.NAME, str(module.VERSION), job)
        if identifier in seen:
            raise StageError(f'{module.NAME}.make_jobs returned the same job twice: {canonical_json(job)}')
        seen.add(identifier)
        entries.append(JobEntry(identifier, job))
    return entries


def select_entries(entries: list[JobEntry], only: Sequence[str] | None) -> list[JobEntry]:
    '''
    Keep the jobs whose ids start with one of the `only` prefixes (all jobs when `only` is empty).
    '''

    if not only:
        return entries
    prefixes = tuple(prefix.lower() for prefix in only)
    chosen = [entry for entry in entries if entry.job_id.startswith(prefixes)]
    if not chosen:
        raise ConfigError(f'no job id starts with {", ".join(only)}')
    return chosen


def resolve_results_root(results_root: str | os.PathLike | None = None, *, smoke: bool = False) -> Path:
    '''
    Results root: the explicit argument, else $LPL_RESULTS_DIR, else ./results. Smoke runs always use a
    ``smoke`` subdirectory of that root, so they never mix with full runs.
    '''

    if results_root is None:
        results_root = os.environ.get(RESULTS_ENV_VAR) or DEFAULT_RESULTS_DIR
    root = Path(results_root)
    return root / 'smoke' if smoke else root


def job_paths(directory: Path, jid: str) -> tuple[Path, Path]:
    '''
    The (arrays, record) file paths of a job.
    '''

    return directory / f'{jid}.npz', directory / f'{jid}.json'


def job_state(directory: Path, jid: str) -> tuple[JobState, dict[str, Any] | None]:
    '''
    Classify a job from its record and arrays. Returns the state and the record when one can be read.
    '''

    npz_path, json_path = job_paths(directory, jid)
    if not json_path.exists():
        return JobState.MISSING, None
    try:
        record = read_json(json_path)
    except (OSError, ValueError):
        return JobState.INCOMPLETE, None
    if not isinstance(record, dict) or record.get('job_id') != jid:
        return JobState.INCOMPLETE, None
    if record.get('status') == 'failed':
        return JobState.FAILED, record
    if record.get('status') != 'ok':
        return JobState.INCOMPLETE, record
    try:
        load_arrays(npz_path)
    except Exception:  # missing, truncated or corrupt archive: any failure means the job is not complete
        return JobState.INCOMPLETE, record
    return JobState.COMPLETE, record


def plan_jobs(directory: Path, entries: Sequence[JobEntry], *, force: bool) -> list[PlannedJob]:
    '''
    Decide for each job whether to run it, and why. Complete jobs are skipped unless `force` is set.
    '''

    plan = []
    for entry in entries:
        state, _ = job_state(directory, entry.job_id)
        if state is JobState.COMPLETE and not force:
            action, reason = 'skip', 'complete'
        else:
            action, reason = 'run', 'forced' if state is JobState.COMPLETE else state.value
        plan.append(PlannedJob(entry.job_id, action, reason, entry.job))
    return plan


def to_storable(result: Any) -> dict[str, np.ndarray]:
    '''
    Check a run_job result and convert it to arrays. Python scalars and strings become 0-d arrays.
    '''

    if not isinstance(result, Mapping):
        raise StageError(f'run_job must return a dict, got {type(result).__name__}')
    arrays: dict[str, np.ndarray] = {}
    for key, value in result.items():
        if not isinstance(key, str) or not key or '/' in key or '\\' in key or key in RESERVED_NPZ_KEYS:
            raise StageError(f'invalid result name {key!r}')
        if isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, (bool, int, float, str, np.generic)):
            arr = np.asarray(value)
        else:
            raise StageError(f'result {key!r} has unsupported type {type(value).__name__}')
        if arr.dtype == object:
            raise StageError(f'result {key!r} is an object array, which would need pickling')
        arrays[key] = arr
    return arrays


# ---------------------------------------------------------------------------
# Thread caps, memory guard and worker pool
# ---------------------------------------------------------------------------


def set_thread_caps() -> None:
    '''
    Limit BLAS, OpenMP and numba to one thread, in this process and in the workers it starts.
    '''

    for var in THREAD_ENV_VARS:
        os.environ[var] = '1'


def default_workers() -> int:
    '''
    Two cores are kept for the parent process and the system, with at most eight workers.
    '''

    return max(1, min(8, (os.cpu_count() or 1) - 2))


def available_bytes() -> int:
    '''
    Memory that can be given to new processes without swapping, in bytes.
    '''

    return int(psutil.virtual_memory().available)


def wait_for_memory(min_free_bytes: int, *, poll_s: float = MEMORY_POLL_S) -> bool:
    '''
    Block until at least `min_free_bytes` are available. Returns True if the call had to wait.
    '''

    free = available_bytes()
    if free >= min_free_bytes:
        return False
    logger.warning('low memory: %.2f GB available, below the %.2f GB floor; holding new jobs', free / _GIB, min_free_bytes / _GIB)
    while free < min_free_bytes:
        time.sleep(poll_s)
        free = available_bytes()
    logger.info('memory recovered: %.2f GB available; resuming', free / _GIB)
    return True


def _init_worker() -> None:
    '''
    Process-pool initializer: cap threads, and ignore SIGINT so that only the parent reacts to Ctrl-C.
    '''

    set_thread_caps()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _make_executor(workers: int) -> Executor:
    '''
    A process pool that uses the spawn start method, so workers never inherit BLAS or OpenMP state.
    '''

    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=_init_worker,
    )


def _peak_rss_mb() -> float | None:
    '''
    High-water resident set size of the calling process, in MB (ru_maxrss is in bytes on macOS, KB elsewhere).
    '''

    if resource is None:
        return None  # Current RSS is not a portable substitute for peak RSS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == 'darwin' else peak / 2**10


def _write_record(
        directory: Path,
        *,
        jid: str,
        job: Mapping[str, Any],
        stage: str,
        version: str,
        status: str,
        error: str | None,
        started_at: str,
        wall_time_s: float,
        worker_pid: int | None,
        peak_rss_mb: float | None,
        result_keys: list[str],
    ) -> None:
    '''
    Write the JSON record of a job, atomically.
    '''

    record = {
        'schema': RECORD_SCHEMA,
        'job_id': jid,
        'stage': stage,
        'stage_version': version,
        'job': dict(job),
        'status': status,
        'error': error,
        'result_keys': result_keys,
        'started_at': started_at,
        'finished_at': provenance.utc_now(),
        'wall_time_s': round(wall_time_s, 6),
        'worker_pid': worker_pid,
        'peak_rss_mb': None if peak_rss_mb is None else round(peak_rss_mb, 1),
    }
    atomic_write_json(job_paths(directory, jid)[1], record)


def _execute_job(
        stage_module: str,
        stage_name: str,
        stage_version: str,
        directory: str,
        jid: str,
        job: dict[str, Any],
    ) -> JobOutcome:
    '''
    Worker entry point: run one job, then store its arrays and record.

    Runs in a spawned worker process (or a thread in tests). Exceptions are stored as a failed record,
    so one bad job cannot stop the run.
    '''

    target = Path(directory)
    npz_path, _ = job_paths(target, jid)
    started_at = provenance.utc_now()
    t0 = time.perf_counter()
    status, error, keys = 'ok', None, []
    try:
        arrays = to_storable(importlib.import_module(stage_module).run_job(job))
        atomic_savez(npz_path, arrays)
        keys = sorted(arrays)
    except Exception:
        status, error = 'failed', traceback.format_exc()
    outcome = JobOutcome(jid, status, time.perf_counter() - t0, os.getpid(), _peak_rss_mb(), error)
    _write_record(
        target,
        jid=jid,
        job=job,
        stage=stage_name,
        version=stage_version,
        status=status,
        error=error,
        started_at=started_at,
        wall_time_s=outcome.wall_time_s,
        worker_pid=outcome.worker_pid,
        peak_rss_mb=outcome.peak_rss_mb,
        result_keys=keys,
    )
    return outcome


def _record_crash(directory: Path, module: ModuleType, entry: JobEntry, exc: BaseException) -> JobOutcome:
    '''
    Record a job that never reported back, because its worker died or the pool broke.
    '''

    if isinstance(exc, BrokenProcessPool):
        error = (
            'worker process terminated abruptly (possibly killed by the OS for memory); '
            'the job did not complete and will be retried on the next run'
        )
    else:
        error = ''.join(traceback.format_exception(exc))
    _write_record(
        directory,
        jid=entry.job_id,
        job=entry.job,
        stage=module.NAME,
        version=str(module.VERSION),
        status='failed',
        error=error,
        started_at=provenance.utc_now(),
        wall_time_s=0.0,
        worker_pid=None,
        peak_rss_mb=None,
        result_keys=[],
    )
    return JobOutcome(entry.job_id, 'failed', 0.0, None, None, error)


class _Progress:
    '''
    Counts finished jobs and logs one line per job, with an ETA.
    '''

    def __init__(self, total: int, skipped: int, to_run: int) -> None:
        self.total = total
        self.skipped = skipped
        self.to_run = to_run
        self.done = 0
        self.failed_ids: list[str] = []
        self._t0 = time.monotonic()

    @property
    def failed(self) -> int:
        return len(self.failed_ids)

    def record(self, outcome: JobOutcome) -> None:
        self.done += 1
        if outcome.status != 'ok':
            self.failed_ids.append(outcome.job_id)
        rss = 'n/a' if outcome.peak_rss_mb is None else f'{outcome.peak_rss_mb:.0f}MB'
        logger.info(
            '[%d/%d] %s %s %.2fs pid=%s rss=%s | skipped=%d failed=%d eta=%s',
            self.skipped + self.done,
            self.total,
            outcome.status.upper(),
            outcome.job_id,
            outcome.wall_time_s,
            outcome.worker_pid,
            rss,
            self.skipped,
            self.failed,
            self._eta(),
        )
        if outcome.error:
            logger.error('job %s failed: %s', outcome.job_id, _last_line(outcome.error))

    def _eta(self) -> str:
        if self.done == 0:
            return '?'
        elapsed = time.monotonic() - self._t0
        return str(timedelta(seconds=round((self.to_run - self.done) * elapsed / self.done)))


def _last_line(text: str | None) -> str:
    lines = [line for line in (text or '').strip().splitlines() if line.strip()]
    return lines[-1] if lines else ''


def _execute(
        module: ModuleType,
        directory: Path,
        entries: Sequence[JobEntry],
        *,
        workers: int,
        min_free_bytes: int,
        progress: _Progress,
    ) -> None:
    '''
    Run `entries` in a process pool, with at most `workers` jobs in flight and a memory check before each
    submission. A job that raises is recorded as failed by its worker. If a worker process dies, the pool
    is rebuilt, and the jobs in flight at that moment are recorded as failed. Failures are counted on `progress`.
    '''

    queue: deque[JobEntry] = deque(entries)
    pending: dict[Future, JobEntry] = {}
    executor = _make_executor(workers)
    pool_broken = False
    try:
        while queue or pending:
            if pool_broken and not pending:
                executor.shutdown(wait=False, cancel_futures=True)
                executor = _make_executor(workers)
                pool_broken = False
            while queue and not pool_broken and len(pending) < workers:
                wait_for_memory(min_free_bytes)
                entry = queue.popleft()
                try:
                    future = executor.submit(
                        _execute_job,
                        module.__name__,
                        module.NAME,
                        str(module.VERSION),
                        str(directory),
                        entry.job_id,
                        entry.job,
                    )
                except BrokenProcessPool:
                    queue.appendleft(entry)
                    pool_broken = True
                    break
                pending[future] = entry
            if not pending:
                continue
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                entry = pending.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:
                    pool_broken = pool_broken or isinstance(exc, BrokenProcessPool)
                    outcome = _record_crash(directory, module, entry, exc)
                progress.record(outcome)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------------------
# Run bookkeeping: logs and manifests
# ---------------------------------------------------------------------------


def _run_paths(root: Path, stage: str) -> tuple[Path, Path, str]:
    '''
    Manifest path, log path and run id for a run. They share one UTC timestamp, with a suffix if another run
    already claimed it.
    '''

    base = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id, suffix = base, 1
    while (root / 'manifests' / f'{stage}-{run_id}.json').exists() or (root / stage / 'logs' / f'run-{run_id}.log').exists():
        suffix += 1
        run_id = f'{base}-{suffix}'
    return root / 'manifests' / f'{stage}-{run_id}.json', root / stage / 'logs' / f'run-{run_id}.log', run_id


@contextmanager
def _run_logging(log_path: Path) -> Iterator[None]:
    '''
    Send the runner's log lines to stderr and to `log_path` for the length of a run.
    '''

    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handlers = [logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler(sys.stderr)]
    previous_level, previous_propagate = logger.level, logger.propagate
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def _exclude_paths(root: Path, project: Path) -> list[str]:
    '''
    Repository-relative results path, so that generated results do not mark the tree as dirty.
    '''

    try:
        relative = root.resolve().relative_to(project)
    except ValueError:
        return []
    return [] if relative == Path('.') else [relative.as_posix()]


def _start_manifest(
        *,
        run_id: str,
        prepared: _Prepared,
        smoke: bool,
        root: Path,
        workers_requested: int,
        workers: int,
        min_free_gb: float,
    ) -> dict[str, Any]:
    project = provenance.project_root()
    return {
        'schema': MANIFEST_SCHEMA,
        'run_id': run_id,
        'stage': prepared.module.NAME,
        'stage_version': str(prepared.module.VERSION),
        'command': list(sys.argv),
        'config': {'path': str(prepared.loaded.path) if prepared.loaded.path else None, 'sha256': prepared.loaded.sha256},
        'merged_config': prepared.cfg,
        'smoke': smoke,
        'results_root': str(root.resolve()),
        'workers': {'requested': workers_requested, 'effective': workers},
        'min_free_gb': min_free_gb,
        'thread_caps': {var: '1' for var in THREAD_ENV_VARS},
        'environment': provenance.environment(project, exclude=_exclude_paths(root, project)),
        'status': 'running',
        'started_at': provenance.utc_now(),
        'finished_at': None,
        'counts': None,
        'failed_job_ids': [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_stage(
        stage: str | ModuleType,
        config: str | os.PathLike | Mapping[str, Any],
        *,
        results_root: str | os.PathLike | None = None,
        workers: int | None = None,
        force: bool = False,
        dry_run: bool = False,
        smoke: bool = False,
        min_free_gb: float = DEFAULT_MIN_FREE_GB,
        only: Sequence[str] | None = None,
    ) -> RunReport:
    '''
    Run the jobs of `stage` for `config` that are not already complete.

    `stage` is a registered name or an importable stage module. `config` is a TOML path or a mapping.
    `force` re-runs complete jobs. `dry_run` only plans. `smoke` applies the SMOKE and [smoke] overrides and
    writes under ``{results}/smoke``. `only` keeps the jobs whose ids start with one of the given prefixes.
    `min_free_gb` holds new jobs while less memory than that is available.

    This sets the thread caps in os.environ for this process and its workers. A script that calls it must
    guard the call with ``if __name__ == '__main__':``, because spawned workers re-import the main module.
    '''

    min_free_bytes = min_free_gb * _GIB
    if min_free_bytes < 0 or not math.isfinite(min_free_bytes):
        raise ConfigError('--min-free-gb must be finite, nonnegative and convertible to bytes')

    prepared = _prepare(stage, config, smoke=smoke)
    module, entries = prepared.module, select_entries(prepared.entries, only)
    root = resolve_results_root(results_root, smoke=smoke)
    directory = root / module.NAME
    plan = plan_jobs(directory, entries, force=force)
    to_run = [entry for entry, item in zip(entries, plan, strict=True) if item.action == 'run']
    skipped = len(entries) - len(to_run)
    if dry_run:
        return RunReport(module.NAME, directory, tuple(plan), len(entries), skipped, 0, (), 'dry-run')

    requested = default_workers() if workers is None else workers
    if requested < 1:
        raise ConfigError('--workers must be at least 1')
    effective = min(requested, len(to_run))
    manifest_path, log_path, run_id = _run_paths(root, module.NAME)
    set_thread_caps()
    manifest = _start_manifest(
        run_id=run_id,
        prepared=prepared,
        smoke=smoke,
        root=root,
        workers_requested=requested,
        workers=effective,
        min_free_gb=min_free_gb,
    )
    progress = _Progress(len(entries), skipped, len(to_run))
    status = 'error'  # replaced by ok, failed or interrupted; an unexpected exception keeps 'error'
    with _run_logging(log_path):
        logger.info(
            'stage %s v%s: %d jobs selected, %d complete (skipped), %d to run | workers=%s min_free=%.1f GB | results in %s',
            module.NAME,
            module.VERSION,
            len(entries),
            skipped,
            len(to_run),
            effective if to_run else '-',
            min_free_gb,
            directory,
        )
        atomic_write_json(manifest_path, manifest)
        try:
            if to_run:
                _execute(
                    module,
                    directory,
                    to_run,
                    workers=effective,
                    min_free_bytes=int(min_free_bytes),
                    progress=progress,
                )
            status = 'ok' if not progress.failed_ids else 'failed'
        except KeyboardInterrupt:
            status = 'interrupted'
            raise
        finally:
            manifest.update(
                status=status,
                finished_at=provenance.utc_now(),
                counts={
                    'total': len(entries),
                    'planned': len(to_run),
                    'run': progress.done,
                    'skipped': skipped,
                    'failed': progress.failed,
                },
                failed_job_ids=progress.failed_ids,
            )
            atomic_write_json(manifest_path, manifest)
            logger.info(
                'finished (%s): %d run, %d skipped, %d failed; manifest %s',
                status,
                progress.done,
                skipped,
                progress.failed,
                manifest_path,
            )
    return RunReport(
        stage=module.NAME,
        results_dir=directory,
        plan=tuple(plan),
        total=len(entries),
        skipped=skipped,
        executed=progress.done,
        failed=tuple(progress.failed_ids),
        status=status,
        manifest_path=manifest_path,
        log_path=log_path,
    )


def _complete_records(
        directory: Path,
        entries: Sequence[JobEntry],
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    states: dict[str, int] = {}
    for entry in sorted(entries, key=lambda item: item.job_id):
        state, _ = job_state(directory, entry.job_id)
        states[state.value] = states.get(state.value, 0) + 1
        if state is JobState.COMPLETE:
            records.append((entry.job, load_arrays(job_paths(directory, entry.job_id)[0])))
    return records, states


def load_records(
        stage: str | ModuleType,
        config: str | os.PathLike | Mapping[str, Any],
        *,
        results_root: str | os.PathLike | None = None,
        smoke: bool = False,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    '''
    The (job, result) pairs of every complete job of `config`, sorted by job id.
    '''

    prepared = _prepare(stage, config, smoke=smoke)
    directory = resolve_results_root(results_root, smoke=smoke) / prepared.module.NAME
    records, _ = _complete_records(directory, prepared.entries)
    return records


def summarize_stage(
        stage: str | ModuleType,
        config: str | os.PathLike | Mapping[str, Any],
        *,
        results_root: str | os.PathLike | None = None,
        out_dir: str | os.PathLike | None = None,
        smoke: bool = False,
    ) -> Path:
    '''
    Call the stage's summarize(records, cfg, out_dir) on the complete records of `config`.

    `out_dir` defaults to ``{results}/{stage}/summary``. Incomplete jobs are left out with a warning.
    Returns the output directory.
    '''

    prepared = _prepare(stage, config, smoke=smoke)
    module = prepared.module
    summarize = getattr(module, 'summarize', None)
    if summarize is None:
        raise StageError(f'stage {module.NAME!r} does not define summarize()')
    directory = resolve_results_root(results_root, smoke=smoke) / module.NAME
    records, _ = _complete_records(directory, prepared.entries)
    if len(records) < len(prepared.entries):
        logger.warning(
            '%d of %d jobs are not complete; summarizing the %d complete records',
            len(prepared.entries) - len(records),
            len(prepared.entries),
            len(records),
        )
    target = Path(out_dir) if out_dir is not None else directory / 'summary'
    target.mkdir(parents=True, exist_ok=True)
    summarize(records, prepared.cfg, target)
    return target


def status_stage(
        stage: str | ModuleType,
        config: str | os.PathLike | Mapping[str, Any],
        *,
        results_root: str | os.PathLike | None = None,
        smoke: bool = False,
    ) -> StatusReport:
    '''
    Count the complete, failed, incomplete and missing jobs of `config`, and list the failures.
    '''

    prepared = _prepare(stage, config, smoke=smoke)
    directory = resolve_results_root(results_root, smoke=smoke) / prepared.module.NAME
    counts = {state.value: 0 for state in JobState}
    failed: list[tuple[str, str]] = []
    for entry in prepared.entries:
        state, record = job_state(directory, entry.job_id)
        counts[state.value] += 1
        if state is JobState.FAILED:
            failed.append((entry.job_id, _last_line((record or {}).get('error'))))
    return StatusReport(
        stage=prepared.module.NAME,
        version=str(prepared.module.VERSION),
        results_dir=directory,
        total=len(prepared.entries),
        counts=counts,
        failed=tuple(sorted(failed)),
    )
