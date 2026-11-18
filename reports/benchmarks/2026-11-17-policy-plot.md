# Policy-plot CLI benchmark, November 17, 2026

This is a bounded local comparison of the `lplearn plot-policy` workflow on
an Apple M1 Max MacBook Pro (10 CPU cores, 64 GB RAM), macOS ARM64, Python
3.12.6. It is not a convergence study or a manuscript reproduction. The
recorded machine load at the end of the runs ranged from approximately 4.3
to 6.5 on the one-minute load average; unrelated system activity could not be
independently excluded.

**Validity of the plotted iterates:** the SGDA outputs measured here are
infeasible for the representative-arm LP relaxation. For K5/T10, 81 of 240
coordinates were below `-0.01`, the minimum was `-0.46035`, and total mass by
stage ranged from `-0.295` to `0.804`. For K11/T30, 666 of 1,440 coordinates
were below `-0.01`, the minimum was `-0.498998`, and stage mass ranged from
`-1.659` to `0.851`. A feasible occupation has nonnegative entries and unit
stage mass. Numba and Rust produced matching iterates, but their agreement does
not make them feasible policies. The then-current plotter hid values below
`0.01`; the saved figures should be read only as diagnostic renderings of
infeasible iterates, not as valid policies.

The tested wheel was built from PR 4 commit `ecd53fb` and installed with its
dependencies in an isolated virtual environment. Its SHA-256 is
`88cf86c6e10e65acf6aee6285ccc1b714373159305432f45057ccc6658b631af`.
The harness and workload configurations are from PR 5 commit `f84f4b8`.
The environment lock lists the installed distribution names and versions;
its SHA-256 is `e50501150bd2f663d8bad98340ba69de15bdf3b555176ba287e8c50044bece8c`.
The lock does not independently verify every third-party package byte.

The workload inputs are `configs/benchmark-k5t10.toml` and
`configs/benchmark-k11t30.toml` at `f84f4b8`. For each input and cache mode,
the measured command had this form (with absolute paths supplied for the
environment, wheel, lock, cache and output):

The historical command below ran against `f84f4b8`, before the feasibility
guard was added. To run the current CLI on these inputs, pass
`--allow-infeasible` to the benchmark runner; it labels such plots explicitly.
The added diagnostics change the work performed, so a rerun is a new measurement.

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$DATABASE_DIR/mpl" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$LPL_VENV/bin/python" scripts/benchmark_policy.py \
  --command "$LPL_VENV/bin/lplearn" --environment-python "$LPL_VENV/bin/python" \
  --cwd "$PWD" --config "$CONFIG" --output-dir "$DATABASE_DIR/$RUN_ID" \
  --case lp:lp:cvxpy --case numba:sgda:numba --case rust:sgda:rust \
  --wheel "$LPL_WHEEL" --lock "$LPL_LOCK" --cpu-model 'Apple M1 Max' \
  --numba-cache-state "$CACHE_MODE" --repetitions 6 \
  --timeout 300 --rust-timeout 240 --mode definitive
```

For `CACHE_MODE=prewarmed`, the command also includes `--numba-cache-dir`
pointing to the cache populated by a preceding call with the same model and
solver settings. For the cold run, `CACHE_MODE=per-process-empty`.

Each run used a single CLI process at a time, one BLAS/OpenMP thread, six
balanced repetitions of each case and 1,000 SGDA iterations. The Numba/Rust
pair had identical model, seed, initialization, sample count and averaging
settings. The LP uses CVXPY/CLARABEL and is a different algorithm. Each command
solves the configured problem, saves its occupation-measure plot, array and
manifest, and exits. Wall time includes startup and plot generation; Rust time
also includes its worker process and JSON transport.

| Model | Numba cache | LP/CVXPY wall median | SGDA/Numba wall median | SGDA/Rust wall median | Paired Numba/Rust wall ratio, median [95% bootstrap interval] |
|---|---|---:|---:|---:|---:|
| K5/T10 | Empty per process | 1.126 s | 5.506 s | 1.171 s | 4.64 [4.07, 5.15] |
| K5/T10 | Prewarmed | 1.290 s | 1.502 s | 1.395 s | 1.08 [0.92, 1.17] |
| K11/T30 | Empty per process | 1.662 s | 6.572 s | 1.875 s | 3.43 [3.14, 3.63] |
| K11/T30 | Prewarmed | 1.608 s | 2.168 s | 1.930 s | 1.09 [0.97, 1.23] |

The ratio is the median of paired Numba wall time divided by Rust wall time;
it is not the ratio of the displayed medians. A value above one favors Rust.
These intervals have only six matched pairs and reflect this machine and these
short workloads. The prewarmed cache was populated using the same configuration
and remained unchanged during each warm run; its presence alone does not prove
that all possible Numba specializations were already compiled. The LP and SGDA
times are reported separately because they solve by different methods.

All 72 CLI commands completed and every corresponding Numba/Rust pair passed
`atol=rtol=2e-11`. Maximum absolute occupation differences were `1.62e-14`
for K5/T10 and `1.55e-14` for K11/T30. All four terminal records report
`complete` with 18/18 jobs, unchanged input identities and one execution
segment. Each run was separately qualified before its measurement. The
runner's interruption/resume test completed independently with ratios suppressed.

| Run | Archived receipt directory | Summary SHA-256 |
|---|---|---|
| K5 cold | `lplearn-pr5-k5-cold-20261117` | `30302d0488640d215bf9242060ade4bf5c8fb7e7f11ca2264e0910eebbc98ac7` |
| K5 warm | `lplearn-pr5-k5-warm-20261117` | `a871cdf592b190ed46b43db4e4c3549dc9c51a32820ab9727e9033d5c0b4ffa6` |
| K11 cold | `lplearn-pr5-k11-cold-20261117` | `d679eb60b55ce7ae677fee25726a6dd45bb095080ee019bdd659bc58ecd4756d` |
| K11 warm | `lplearn-pr5-k11-warm-20261117` | `90b2fe316c65b4b41ca408f882579a1bee2855c571aa71f00a491be00057808e` |

Each directory contains `provenance.json`, `progress.json`, `terminal.json`,
`summary.json`, `receipts.jsonl`, hashed shards, and the generated plots and
arrays. The maintainers retained the run directories in a verified external
archive. See the [benchmark protocol](../../docs/benchmark-protocol.md) for
the command, reproducibility gates, and timing boundaries.
