"""Export browser model and reference JSON from supplied experiment summaries.

Outputs default to the runner's summary directory. Nothing is written to a
frontend checkout unless paths there are explicitly selected in the config.
"""

from pathlib import Path

from lplearn.experiments.io import atomic_write_json
from lplearn.experiments.runner import StageError

NAME = 'web'
VERSION = '2'  # summary-local output, validation and data-derived provenance
SMOKE = {'results_root': 'results/smoke', 'commons_summary_dir': 'commons/summary'}


def make_jobs(cfg):
    return [{'n_files': 2}]


def run_job(job):
    return {'n_files': 2}


def summarize(records, cfg, out_dir):
    from lplearn.visualizations import tutorial_assets, web_export

    out_dir = Path(out_dir)
    try:
        model, reference = web_export.build_web_data(cfg)
        model_path = Path(cfg.get('model_json', out_dir / 'model.json'))
        reference_path = Path(cfg.get('reference_json', out_dir / 'reference.json'))
        manifest_path = out_dir / 'web_manifest.json'
        if model_path.suffix.lower() != '.json' or reference_path.suffix.lower() != '.json':
            raise ValueError('model and reference outputs must use the .json extension')
        paths = [p.resolve() for p in (model_path, reference_path, manifest_path)]
        if len(set(paths)) != 3:
            raise ValueError('model, reference and manifest must use distinct output paths')
        sources = {Path(p).resolve() for p in model['inputs']}
        if any(p in sources for p in paths):
            raise ValueError('an output path would overwrite an input summary')
        model_path = web_export.write_json(model_path, model)
        reference_path = web_export.write_json(reference_path, reference)
        atomic_write_json(manifest_path, {
            'written': [str(model_path), str(reference_path)], 'inputs': model['inputs'],
            'grid': [[g['alpha'], g['beta']] for g in model['lpGrid']], 'lp_value': model['results']['lpValue'],
            'replication_counts': model['results']['replicationCounts'],
        })
    except (tutorial_assets.TutorialInputError, ValueError, KeyError, TypeError, OSError) as exc:
        raise StageError(f'web: {exc}') from exc
