'''
Provenance for experiment runs: which code, config and machine produced a set of results.

Only read-only commands are run (git rev-parse, git status, git diff, sysctl).
'''

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

PACKAGES = ('lplearn', 'numpy', 'scipy', 'cvxpy', 'clarabel', 'numba', 'pandas', 'matplotlib', 'psutil')
MAX_LISTED_PATHS = 200


def utc_now() -> str:
    '''
    Current UTC time as an ISO 8601 string (seconds resolution).
    '''

    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256_bytes(data: bytes) -> str:
    '''
    Hex SHA-256 digest of `data`.
    '''

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    '''
    Hex SHA-256 digest of the file at `path`, or None if there is no such file.
    '''

    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def project_root(start: Path | None = None) -> Path:
    '''
    Nearest directory at or above `start` (default: the working directory) that contains pyproject.toml.
    '''

    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / 'pyproject.toml').is_file():
            return candidate
    return start


def environment(root: Path, exclude: Sequence[str] = ()) -> dict[str, Any]:
    '''
    Caller working-tree state and installed runtime identity for a run.

    `exclude` holds repository-relative paths (such as the results directory) whose changes do not count as dirty.
    '''

    return {
        # The caller's checkout can differ from the installed code actually
        # imported by Python. Never present its commit as the runtime identity.
        'caller_project': {
            'root': str(root.resolve()),
            'git': git_state(root, exclude),
            'uv_lock_sha256': sha256_file(root / 'uv.lock'),
            'source_tree': source_tree(root),
        },
        'installed_runtime': installed_runtime(),
        'python': {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'executable': sys.executable,
        },
        'platform': {
            'platform': platform.platform(),
            'macos_version': platform.mac_ver()[0] or None,
            'machine': platform.machine(),
        },
        'cpu': {'model': cpu_model(), 'logical_cores': os.cpu_count()},
        'ram_total_gb': round(psutil.virtual_memory().total / 2**30, 2),
        'packages': package_versions(),
    }


def installed_runtime() -> dict[str, Any]:
    '''Identify imported Python sources and packaged workers without inferring a Git commit.

    Hash the source files from the imported package's actual location, including
    editable installs, and locate optional native workers using the distribution
    RECORD in the same manner as the Python adapters. Listing a worker identifies
    an available binary; it does not assert that a particular job executed it.
    '''
    import lplearn

    module_path = Path(lplearn.__file__).resolve()
    package_root = module_path.parent
    files = sorted(path for path in package_root.rglob('*') if path.is_file()
                   and path.suffix in ('.py', '.so', '.pyd', '.dll', '.dylib'))
    digest = hashlib.sha256()
    file_hashes = {}
    for path in files:
        relative = path.relative_to(package_root).as_posix()
        value = sha256_file(path)
        file_hashes[relative] = value
        digest.update(relative.encode('utf-8') + b'\0' + value.encode('ascii') + b'\0')

    result = {
        'import_path': str(module_path),
        'package_root': str(package_root),
        'runtime_files_sha256': digest.hexdigest(),
        'runtime_file_count': len(files),
        'runtime_files': file_hashes,
        'distribution_version': None,
        'distribution_record_path': None,
        'distribution_record_sha256': None,
        'distribution_matches_import': None,
        'packaged_workers': {},
    }
    try:
        dist = importlib.metadata.distribution('lplearn')
    except importlib.metadata.PackageNotFoundError:
        return result
    result['distribution_version'] = dist.version
    result['distribution_matches_import'] = Path(dist.locate_file('lplearn/__init__.py')).resolve() == module_path
    for entry in dist.files or ():
        if entry.name == 'RECORD' and entry.parent.name.endswith('.dist-info'):
            record_path = Path(dist.locate_file(entry)).resolve()
            result['distribution_record_path'] = str(record_path)
            result['distribution_record_sha256'] = sha256_file(record_path)
        if entry.name in ('lplearn-native', 'lplearn-native.exe', 'lplearn-solver', 'lplearn-solver.exe'):
            worker = Path(dist.locate_file(entry)).resolve()
            result['packaged_workers'][entry.name] = {'path': str(worker), 'sha256': sha256_file(worker)}
    return result


SOURCE_GLOBS = ('python/lplearn/**/*.py', 'src/**/*.rs', 'configs/*.toml', 'scripts/*.py',
                'pyproject.toml', 'Cargo.toml', 'Cargo.lock', 'uv.lock')


def source_tree(root: Path) -> dict[str, Any]:
    '''
    SHA-256 over the paths and contents of every file that can change results (package sources, configs, scripts,
    dependency files), tracked by git or not. Unlike `git diff HEAD`, it covers untracked files.
    '''

    files = sorted({path for pattern in SOURCE_GLOBS for path in root.glob(pattern) if path.is_file()})
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return {'sha256': digest.hexdigest(), 'n_files': len(files), 'globs': list(SOURCE_GLOBS)}


def git_state(root: Path, exclude: Sequence[str] = ()) -> dict[str, Any]:
    '''
    Read-only description of the git working tree at `root`: commit, branch, dirty paths, and a hash of `git diff HEAD`.
    '''

    commit = _git_output(root, ['rev-parse', 'HEAD'])
    if commit is None:
        return {'commit': None, 'branch': None, 'dirty': None, 'dirty_paths': [], 'diff_sha256': None}

    pathspec = ['--', '.', *(f':(exclude){path}' for path in exclude)]
    status = (_git_output(root, ['status', '--porcelain', *pathspec]) or b'').decode('utf-8', 'replace')
    diff = _git_output(root, ['--no-pager', 'diff', '--no-color', '--no-ext-diff', 'HEAD', *pathspec]) or b''
    branch = _git_output(root, ['rev-parse', '--abbrev-ref', 'HEAD']) or b''
    dirty_paths = status.splitlines()
    return {
        'commit': commit.decode().strip(),
        'branch': branch.decode().strip(),
        'dirty': bool(dirty_paths),
        'dirty_paths': dirty_paths[:MAX_LISTED_PATHS],
        'diff_sha256': sha256_bytes(diff),
    }


def cpu_model() -> str | None:
    '''
    CPU brand string (sysctl on macOS), falling back to platform.processor().
    '''

    out = _run(['sysctl', '-n', 'machdep.cpu.brand_string'])
    if out and out.strip():
        return out.decode('utf-8', 'replace').strip()
    return platform.processor() or None


def package_versions(names: Sequence[str] = PACKAGES) -> dict[str, str | None]:
    '''
    Installed versions of `names` (None for packages that are not installed).
    '''

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_output(root: Path, args: Sequence[str]) -> bytes | None:
    return _run(['git', *args], cwd=root)


def _run(args: Sequence[str], cwd: Path | None = None) -> bytes | None:
    '''
    stdout of a read-only command, or None if it is unavailable or exits with an error.
    '''

    env = {**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}  # do not refresh the git index as a side effect
    try:
        done = subprocess.run(list(args), cwd=cwd, env=env, capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout
