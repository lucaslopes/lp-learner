# Experiment runner

`lplearn experiments` ports the experiment runner from the original Python
reference (see `NOTICE`). It executes registered jobs in a spawned process
pool, saves one compressed array archive and JSON record per job, and skips
complete jobs when rerun. Use `--results-dir` or `LPL_RESULTS_DIR` to place
generated results outside the source tree.

```sh
lplearn experiments list
lplearn experiments run smoke --config configs/smoke.toml --workers 1 --min-free-gb 0 --results-dir /tmp/lplearn-results
lplearn experiments status smoke --config configs/smoke.toml --results-dir /tmp/lplearn-results
lplearn experiments summarize smoke --config configs/smoke.toml --results-dir /tmp/lplearn-results
```

The included `smoke` stage solves small random matrix eigenproblems to exercise
the runner. It is not a queue-policy experiment. The `lp_solutions` and
`learners` stages reproduce the reference experiment grids using Python CVXPY
and either Numba or the packaged Rust backend, respectively. Set
`backend = "rust"` in the learners configuration; the default is `"numba"`.
`rust_timeout` sets the worker timeout per job (default 3600 s). Job IDs include the stage version, selected backend, and the installed Numba
version or Rust worker SHA-256. These fields keep results from different
backends and worker versions separate. Use separate summary
output directories for the two backends. Rust failures mark jobs failed and
never silently fall back to Numba. Both backends retain the same seeds and
state-aggregation weights. Each learner record includes the raw occupation
minimum, flow/budget/nonnegativity violations, backend identity and timing
scope; short runs can be infeasible and these diagnostics do not establish
policy feasibility.

The `bound` stage evaluates the state-aggregation lemmas and spectral bounds
for small research configurations. Its default config includes expensive
spectral grids, so inspect it and use `--smoke` or a deliberately reduced
config before an exploratory run.

The `finite_n` stage simulates projection, LP update or independent policies using
the reference mechanisms and common random numbers. For learned occupation
measures, each job records raw minimum, mass per stage, constraint violations,
and the number of fallback rows and empty stages before policy conversion;
non-finite learned values are rejected. See `configs/finite_n.toml` for inputs.
Commons and paper stages remain unported; selecting one returns a clear error.

The shipped `configs/learners.toml` describes a large research grid, including
millions of iterations per job. Inspect its parameters and use a deliberately
small separate configuration before running an exploratory test. Do not assume
that selecting `--smoke` makes the reference workload trivial.
The finite-N configuration also defines a large grid of replications; use a
separate, small configuration and a writable external results directory for
qualification runs.

The runner defaults to a 4 GiB free-memory threshold and may wait for memory.
`--dry-run`, `--force`, `--only`, and `--smoke` retain their reference meanings.
Each run writes a manifest recording config and environment identities. On
Windows the `resource` module is unavailable, so worker peak RSS is reported
as null rather than as a current-memory measurement.

Manifest schema 2 separates `environment.caller_project` (the working
directory's Git state and source-tree hash) from `environment.installed_runtime`
(the imported package path, hashes of its Python/native module files, the
distribution RECORD hash, and the paths and hashes of packaged workers). The
caller's Git commit is not assumed to be the revision of the imported wheel.
Listing a worker records an available binary; it does not assert that a job
executed that worker. Distribution metadata may be absent or refer to a
different import path for editable or otherwise unusual installations.
