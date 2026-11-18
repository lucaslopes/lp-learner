'''
Atomic file helpers for experiment results.

Every write goes to a uniquely named temporary file in the destination directory
and is moved into place with os.replace, so readers never see a partial file and
an interrupted run leaves at most a stray ``*.tmp`` file behind.
'''

import json
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

import numpy as np

# Keys that np.savez_compressed takes as its own arguments, so results cannot use them.
RESERVED_NPZ_KEYS = frozenset({'file', 'allow_pickle'})


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    '''
    Write `data` to `path` atomically.
    '''

    _atomic_write(Path(path), lambda handle: handle.write(data))


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    '''
    Write UTF-8 `text` to `path` atomically.
    '''

    atomic_write_bytes(path, text.encode('utf-8'))


def atomic_write_json(path: str | os.PathLike, obj: Any) -> None:
    '''
    Write `obj` to `path` as sorted, indented JSON, atomically.
    '''

    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=str)
    atomic_write_text(path, text + '\n')


def atomic_savez(path: str | os.PathLike, arrays: Mapping[str, np.ndarray]) -> None:
    '''
    Save `arrays` as a compressed ``.npz`` at `path`, atomically. Object arrays are refused.
    '''

    clashes = RESERVED_NPZ_KEYS.intersection(arrays)
    if clashes:
        raise ValueError(f'result names {sorted(clashes)} are reserved')
    if any(np.asarray(array).dtype.hasobject for array in arrays.values()):
        raise ValueError('object arrays are not supported')
    # NumPy 1.26 has no allow_pickle parameter for savez_compressed; passing it
    # would silently create an unintended array with that name.
    _atomic_write(Path(path), lambda handle: np.savez_compressed(handle, **arrays))


def read_json(path: str | os.PathLike) -> Any:
    '''
    Parse the JSON file at `path`.
    '''

    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def load_arrays(path: str | os.PathLike) -> dict[str, Any]:
    '''
    Load an archive written by atomic_savez. Pickled data is refused, and 0-d arrays are returned as Python scalars.
    '''

    with np.load(path, allow_pickle=False) as data:
        return {key: _unwrap(data[key]) for key in data.files}


def _unwrap(arr: np.ndarray) -> Any:
    return arr.item() if arr.ndim == 0 else arr


def _atomic_write(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp')
    try:
        with open(tmp, 'wb') as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
