# Web data export

The `web` stage adapts the browser-data generator from `lp-learner-new` at
`c23e6007400daae28737f91d19205233d313d043`. Its frontend is not included.
Run and summarize with `configs/web.toml`, using an external results directory:

```sh
lplearn experiments run web --config configs/web.toml --results-dir /tmp/my-run --workers 1
lplearn experiments summarize web --config configs/web.toml --results-dir /tmp/my-run
```

Configure `results_root` to a directory with the upstream `commons` and
`finite_n` summary CSVs. Six CSVs are required: `commons_summary.csv`,
`commons_paired.csv`, `commons_dilemma.csv`, `commons_stages.csv`,
`commons_fluid.csv` and `finite_n_summary.csv`. Missing rows or inputs fail
with a diagnostic. These are recorded experiment results, not simulations
run by the export. The finite-N LP objective must match the configured base
model. Input paths and SHA-256 hashes are recorded in `model.json` and
`web_manifest.json`. Matching one objective alone cannot establish that all
CSV inputs came from the configured model; independently verify their origin
before using them as scientific results.

By default output is `/tmp/my-run/web/summary/{model.json,reference.json,web_manifest.json}`.
Explicit `model_json` and `reference_json` paths may instead be supplied; they
must be distinct JSON paths and cannot overwrite input CSVs. Each JSON write is
atomic; the pair is not a transaction, so an I/O failure between writes may
leave versions from different runs. Configure `alphas`, `betas`, and `[model]`
to select the grid and model. The grid LP occupations are checked for mass,
flow, nonnegativity and budgets before and after rounding. A GDA iterate in the
reference is explicitly labeled when infeasible. Non-finite outputs fail.

The existing frontend expects one numeric `replications` count. The exporter
requires a single positive count shared by supplied summary tables (zero-count
verdict rows are ignored), rather than labeling arbitrary input as 2,000 runs.
It adds validation and provenance properties; frontend compatibility and real
study data must be validated separately. Synthetic fixtures test behavior only;
they are not scientific results.
