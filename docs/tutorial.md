# Tutorial figures and tables

The `tutorial` stage ports the tutorial asset generator from the Python
reference identified in `NOTICE`. Its complete configuration requires recorded
`commons` summaries (the five CSVs in `commons/summary_tutorial`), the
`finite_n` summary CSV, and, for `tutorial_learned`, the learners solution NPZ.
Set `results_root` in the TOML to the directory containing these inputs. The
runner's `--results-dir` controls jobs and the default summary directory; it
does not change this input root. See `configs/tutorial.toml` for paths. The
upstream stages can be computationally expensive; this port does not claim to
reproduce the historical studies.

By default PDFs and TeX go into the runner's `tutorial/summary/figures` and
`tutorial/summary/tables`, never into the manuscript tree. The configuration's
optional `figure_dir` and `table_dir` override these defaults and can overwrite
existing files at the chosen destination: use fresh directories outside a source
tree. `tutorial_manifest.json` records the LP value, input hashes, files written,
failures, and whether the numbers were requested. Missing or malformed required
inputs cause a nonzero status; data-independent requested outputs can still be
written and are recorded as partial results. Use a fresh directory so prior
outputs cannot be mistaken for the current result.

For a small synthetic test without recorded studies, create `tutorial-small.toml`:

```toml
results_root = "/tmp/lplearn-tutorial-inputs"
figures = ["tutorial_kernel"]
tables = ["parameters"]
write_numbers = false
```

Then run:

```sh
lplearn experiments run tutorial --config tutorial-small.toml --results-dir /tmp/lplearn-tutorial-run --workers 1 --min-free-gb 0
lplearn experiments status tutorial --config tutorial-small.toml --results-dir /tmp/lplearn-tutorial-run
lplearn experiments summarize tutorial --config tutorial-small.toml --results-dir /tmp/lplearn-tutorial-run
```

`write_numbers` defaults to true and requires recorded data and a simulated
illustrative run. Other choices are listed in `configs/tutorial.toml`. The
small example solves the default CJS LP but does not run the full studies or
validate the manuscript's empirical numbers.
