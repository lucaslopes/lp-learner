'''
Stage ``tutorial``: figures, tables and quoted numbers of the undergraduate tutorial (papers/Tutorial/main.tex).

Like ``figures`` and ``tables``, the stage has one trivial job; ``summarize`` redraws everything from the model
parameters and the recorded summaries of the ``commons`` stage (run with configs/commons_tutorial.toml) and of the
``finite_n`` stage, into ``figure_dir`` and ``table_dir``. It writes ``tutorial_manifest.json`` (model, LP value,
input files with hashes, the quoted numbers, failures) into the stage summary directory.

    lplearn experiments run tutorial --config configs/tutorial.toml
    lplearn experiments summarize tutorial --config configs/tutorial.toml

The work is done in lplearn/visualizations/tutorial_assets.py, which is imported lazily.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any

from lplearn.experiments.io import atomic_write_json
from lplearn.experiments.runner import StageError


NAME = 'tutorial'
VERSION = '2'
SMOKE = {'results_root': 'results/smoke'}
MANIFEST_NAME = 'tutorial_manifest.json'


def _assets():
    from lplearn.visualizations import tutorial_assets

    return tutorial_assets


def make_jobs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    '''
    One job that lists the assets to build. Unknown names are refused here, before anything runs.
    '''

    assets = _assets()
    figures = list(cfg.get('figures', assets.FIGURES))
    tables = list(cfg.get('tables', assets.TABLES))
    unknown = [f for f in figures if f not in assets.FIGURES] + [t for t in tables if t not in assets.TABLES]
    if unknown:
        raise StageError(f'unknown tutorial asset(s) {unknown}; known: {", ".join(assets.FIGURES + assets.TABLES)}')
    write_numbers = cfg.get('write_numbers', True)
    if not isinstance(write_numbers, bool):
        raise StageError('write_numbers must be a boolean')
    return [{'figures': figures, 'tables': tables, 'write_numbers': write_numbers}]


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    '''
    Records how many assets the job lists. The building happens in ``summarize``.
    '''

    return {'n_figures': len(job['figures']), 'n_tables': len(job['tables'])}


def summarize(records: list[tuple[dict[str, Any], dict[str, Any]]], cfg: dict[str, Any], out_dir: Path) -> None:
    '''
    Build every requested asset. Raises StageError after building the others if any asset could not be built.
    '''

    effective_cfg = dict(cfg)
    effective_cfg.setdefault('figure_dir', str(Path(out_dir) / 'figures'))
    effective_cfg.setdefault('table_dir', str(Path(out_dir) / 'tables'))
    try:
        result, manifest = _assets().build_assets(effective_cfg)
    except (ValueError, KeyError, TypeError, _assets().TutorialInputError) as exc:
        atomic_write_json(Path(out_dir) / MANIFEST_NAME, {
            'phase': 'preparation', 'written': [], 'failed': {'preparation': str(exc)},
        })
        raise StageError(f'tutorial preparation: {exc}') from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / MANIFEST_NAME, manifest)
    if result.failed:
        details = '; '.join(f'{name}: {message}' for name, message in result.failed.items())
        raise StageError(f'{len(result.failed)} tutorial asset(s) not written ({len(result.written)} written): {details}')
