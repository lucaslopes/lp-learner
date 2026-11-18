"""Small synthetic checks; no recorded manuscript study is required."""
import json
from pathlib import Path

import pytest

from lplearn.experiments.runner import StageError
from lplearn.experiments.stages import tutorial


def test_synthetic_tutorial_outputs_and_reference(tmp_path):
    out = tmp_path / 'run' / 'summary'
    cfg = {'results_root': str(tmp_path / 'missing'), 'figures': ['tutorial_kernel'],
           'tables': ['parameters'], 'write_numbers': False}
    tutorial.summarize([], cfg, out)
    manifest = json.loads((out / 'tutorial_manifest.json').read_text())
    assert manifest['phase'] == 'complete'
    assert manifest['inputs'] == {}
    # Pinned Python reference c23e600, default CJS instance, CLARABEL.
    assert manifest['lp_value'] == pytest.approx(384.25395974391034, abs=1e-5)
    assert (out / 'figures' / 'tutorial_kernel.pdf').read_bytes().startswith(b'%PDF-')
    assert (out / 'tables' / 'parameters.tex').is_file()
    assert not (out / 'tables' / 'numbers.tex').exists()
    assert not (tmp_path / 'papers').exists()


def test_missing_inputs_are_reported_with_partial_outputs(tmp_path):
    out = tmp_path / 'summary'
    cfg = {'results_root': str(tmp_path / 'missing'), 'figures': [],
           'tables': ['parameters', 'results_main'], 'write_numbers': False}
    with pytest.raises(StageError, match='missing input'):
        tutorial.summarize([], cfg, out)
    manifest = json.loads((out / 'tutorial_manifest.json').read_text())
    assert manifest['phase'] == 'partial'
    assert 'missing input' in manifest['failed']['results_main']
    assert (out / 'tables' / 'parameters.tex').is_file()
    assert not (out / 'tables' / 'results_main.tex').exists()


def test_invalid_write_numbers_is_rejected():
    with pytest.raises(StageError, match='boolean'):
        tutorial.make_jobs({'figures': [], 'tables': [], 'write_numbers': 'false'})
