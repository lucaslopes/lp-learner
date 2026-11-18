# Exploratory recovered-policy CLI benchmark

On November 17, 2026, the installed wheel built from `eff4c1e` was measured
with `configs/benchmark-k5t10.toml` (K=5, T=10, 1,000 SGDA steps). Each CLI
run solved the model, rendered the occupation plot, and wrote its arrays and
manifest. Learning cases used `--recover`: a CVXPY/CLARABEL quadratic program
projected the raw iterate onto the representative-arm LP constraints. The
conditional policy was re-evaluated in the model and checked separately. The
CVXPY LP case ran directly without the extra projection.

The raw SGDA outputs were infeasible in all runs. All 12 recovered SGDA
occupations and their induced policies passed the LP-relaxation checks to
`1e-6`. This recovery changes the learned iterate; it does not establish
convergence, optimality, or sample-path feasibility in a finite population.

| Cache | LP/CVXPY wall median | SGDA/Numba + recovery wall median | SGDA/Rust + recovery wall median | Paired Numba/Rust wall ratio [95% bootstrap interval] |
|---|---:|---:|---:|---:|
| Empty per process | 1.236 s | 5.530 s | 1.274 s | 4.36 [4.11, 4.52] |
| Prewarmed | 1.234 s | 1.366 s | 1.260 s | 1.07 [1.04, 1.11] |

These are **exploratory measurements**, not a hardware-general speed claim.
Each scenario has six matched pairs. The host was an Apple M1 Max MacBook Pro
with 10 CPU cores and macOS ARM64; recorded load was elevated (one-minute
load average 9.58 at the end of the cold run and 5.95 at the end of the warm
run), and other system activity could not be independently excluded. Numba's
prewarmed cache is a declared cache state, not a proof that every specialization
was already compiled. The cold Numba time includes JIT compilation. The ratio
is a median of paired wall-time ratios, not the ratio of the displayed medians.
LP and SGDA are different methods and are not given a direct speedup ratio.

All 36 commands passed. The maximum Numba/Rust absolute differences were
`1.62e-14` for the raw iterate, `5.18e-13` for the recovered occupation and
`3.15e-12` for the conditional policy. Raw parity used `atol=rtol=2e-11`;
recovery and policy parity used `atol=rtol=2e-7`. Median projection-solver
times were 7.0 ms (Numba) and 23.8–24.4 ms (Rust); these are separate from
the full CLI wall time and do not measure only the optimization loop.

The benchmark harness was commit `eff4c1e`; the installed wheel SHA-256 was
`3d036e0b57087ff570933033b09d41b1227b6e58f0d7676c535eeb791206a818`.
With that revision and its matching wheel, the measured command had this form:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$DATABASE_DIR/mpl" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$LPL_VENV/bin/python" scripts/benchmark_policy.py \
  --command "$LPL_VENV/bin/lplearn" --environment-python "$LPL_VENV/bin/python" \
  --cwd "$PWD" --config configs/benchmark-k5t10.toml \
  --output-dir "$DATABASE_DIR/$RUN_ID" --case lp:lp:cvxpy \
  --case numba:sgda:numba --case rust:sgda:rust --wheel "$LPL_WHEEL" \
  --lock "$LPL_LOCK" --cpu-model 'Apple M1 Max' --numba-cache-state "$CACHE_MODE" \
  --repetitions 6 --timeout 300 --rust-timeout 240 --recover --mode definitive
```

For the warm run, `$CACHE_MODE` was `prewarmed` and the command additionally
specified `--numba-cache-dir` with a cache populated by a prior matching
`plot-policy` call. For the cold run it was `per-process-empty`.
The cold and warm runs were written to external run directories and include
terminal, provenance, receipt and summary records. A hash-verified local copy
includes the runs, wheel and lock; its long-term retention is not established.
