# Publication figures from experiment results

The `figures` stage adapts the publication figure generator and settings from
the Python reference at commit `c23e6007400daae28737f91d19205233d313d043`.
It reads the numeric `.npz` and `.csv` outputs of the `lp_solutions`,
`learners`, `bound`, `finite_n` and `commons` stages. See the figure/input
mapping in `configs/figures.toml` and the `FILE_SCHEMAS` definitions in
`lplearn.visualizations.paper_figures`. Supplying a subset in `figures` only
requires inputs for that subset. The original manuscript and its historical
results are not included in this package.

Use a separate results directory and output directory when evaluating or
reproducing figures. For example, after editing a copy of `figures.toml` to
set `results_root`, `out_dir` and `figures`:

```sh
lplearn experiments run figures --config figures.toml --results-dir /path/to/records --workers 1
lplearn experiments summarize figures --config figures.toml --results-dir /path/to/records
```

`run` records the requested figure names; `summarize` generates the figures
and writes `figures_manifest.json` in the stage summary. Each successful PDF
is saved atomically and the manifest records its hash and input hashes. A
failure is recorded, the stage returns an error, and any pre-existing PDF is
preserved; such PDFs appear in the manifest under `preserved` and are not
outputs of the failed run. Select a fresh output directory when a consumer
cannot distinguish preserved files from new outputs.

Pie figures follow the reference's action order and display threshold of 0.01;
they clip and renormalize state masses for visualization. A panel with
materially negative values or non-unit stage mass is labeled **Infeasible
iterate (clipped)**. The check does not verify model flow or resource
constraints, so absence of that label does not certify feasibility. The
existing legacy helper also emits a warning when these basic checks fail.
The plots do not modify stored source arrays. Missing scientific inputs are
errors, not permission to substitute synthetic data.

`tests/pytest_figure_checks.py` uses synthetic records to check every figure,
the action order, PDF fonts and repeatability, stage errors, and preservation
of existing output on failure. These tests validate the code path; they do not
reproduce the manuscript's numerical results or validate figures in a paper.
