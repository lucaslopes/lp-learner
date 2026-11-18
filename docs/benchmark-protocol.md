# Policy plot benchmark protocol

Run the same policy-plot CLI in separate processes, once per case and repetition.
Every case receives the same frozen TOML, varies only method/backend and output,
and yields a plot, numeric occupation array, and CLI manifest. The harness checks
artifact hashes, config identity and finite arrays, retains failures, and records
external command wall time. Case order is rotated within shuffled blocks. Each
committed job has an atomic, content-hashed shard and a unique attempt directory;
`progress.json` provides a heartbeat, phase, completed jobs, timing and load.
Collection, aggregation and terminal verification are separate phases.

From the repository root, after installing a locally built wheel in a Python
3.12 virtual environment, set `LPL_VENV` to its absolute path, `LPL_WHEEL` to
the installed wheel, and `DATABASE_DIR` to a writable directory outside the
repository. Run a disposable qualification with the same Python and executable
that will be measured:

```bash
run_qualification() {
"$LPL_VENV/bin/python" scripts/benchmark_policy.py \
  --command "$LPL_VENV/bin/lplearn" --cwd "$PWD" \
  --config configs/policy.toml --output-dir "$DATABASE_DIR/benchmark-qualification" \
  --case lp:lp:cvxpy --case numba:sgda:numba --case rust:sgda:rust \
  --environment-python "$LPL_VENV/bin/python" --numba-cache-state per-process-empty \
  --wheel "$LPL_WHEEL" --allow-infeasible --repetitions 1 "$@"
}
run_qualification --stop-after 1  # exit 130 at a committed boundary
run_qualification --resume        # complete the remaining jobs
run_qualification --resume        # independently verify terminal artifacts
```

This qualification checks the installed CLI, artifacts, and terminal reload;
it does not produce a performance comparison. `configs/policy.toml` is the
small fixture. `configs/benchmark-k5t10.toml` and
`configs/benchmark-k11t30.toml` specify larger fixed budgets for separate runs.
They are short timing workloads, not evidence of convergence or reproductions
of manuscript experiments. For GDA replace both `sgda` case values with `gda`.

For same-method pairs with different backends, `summary.json` records numerical
parity and checks that the resolved configuration, seed, iterations, sample
count and averaging count match. A wall-time ratio and paired bootstrap interval
are reported only in definitive runs when at least five complete pairs pass
both checks and every requested case succeeds. Qualification runs suppress all
ratios, including individual pairs. CVXPY LP and SGDA use different methods,
so no speed ratio is calculated for those. `receipts.jsonl` records each
invocation's command, process exit, stdout/stderr, wall time and validated CLI
manifest.

The plotting command checks occupation feasibility and refuses infeasible
iterates by default. To benchmark their diagnostic rendering, explicitly
provide `--allow-infeasible` to the benchmark runner. It forwards the flag
uniformly to every case, records it in the run identity, checks that each
infeasible figure is labeled as a diagnostic, and reports counts of feasible
and infeasible results. Timing an infeasible iterate does not validate a policy.

Alternatively, pass `--recover` to benchmark a plot of a recovered policy. The
runner forwards it **only** to GDA/SGDA cases. The LP/CVXPY case remains the
direct LP solution, without an additional projection, and is reported as a
different method. Recovery projects the learned occupation onto the
representative-arm LP constraints using CVXPY/CLARABEL; it does not establish
learner convergence, optimality or finite-population sample-path feasibility.
`--recover` and `--allow-infeasible` are mutually exclusive.

For recovered cases, the benchmark independently rebuilds the selected model,
checks the raw and recovered occupation arrays and their hashes, verifies the
derived conditional policy is a row-stochastic normalization of the recovered
occupation, and evaluates its flow and expected resource constraints. It also
compares the original iterates, recovered occupations, and conditional policies
between paired learning backends. Raw parity uses `--atol`/`--rtol`; projected
occupations and policies use `--recovery-atol`/`--recovery-rtol` (default 2e-7).
If any paired output fails, speed ratios are suppressed. The summary's legacy
`feasible` field means the *raw* occupation was feasible; `raw_occupation_feasible`,
`recovered`, and `plotted_relaxation_feasible` make this distinction explicit.

Wall time includes model setup, Rust transport where applicable, recovery,
verification and plot/file output. `median_solve_seconds` excludes recovery;
`median_projection_solve_seconds` reports the CLARABEL call reported by the
CLI and excludes setup, validation and policy construction. The direct LP and
recovered learning pipelines are different methods; do not interpret their
times as interchangeable backend implementations.

**Timing boundary:** the full wall clock covers process startup, loading,
model setup, solver and plot writing. The CLI manifest reports solve and plot
times. `--numba-cache-state per-process-empty` assigns an empty Numba cache to
each process. Alternatively `--numba-cache-state prewarmed --numba-cache-dir DIR`
requires a pre-populated cache directory; this is an assertion about cache state,
not proof that the relevant specialization was compiled. Neither mode reports
pure kernel time. Compare cold and warm configurations before drawing conclusions.
Use a new output directory, outside the source working directory, for each run.
To recover an incomplete run, repeat the identical command with `--resume`.
The runner verifies its source, configuration, wheel, lock, CLI, environment,
run identity, existing shards and artifact hashes before skipping committed
jobs. Partial attempts are preserved and recorded as orphans; resumption stops
if an orphan's worker state is unknown or its process group might still run.
On Windows, an outer CLI timeout cannot prove that child solver processes have
stopped. The run stops immediately with an uncommitted attempt, and automatic
resumption is refused. Inspect and stop any remaining solver processes before
starting a new run in a separate output directory. This conservative behavior
does not terminate the full process tree for the user.
The output includes atomic `summary.json` and `terminal.json`; all shards and
aggregate artifacts are reloaded before the terminal state is marked complete.
`--stop-after N` is a qualification-only way to stop at a committed job boundary
and exercise recovery. Runs with multiple execution segments suppress their
performance ratio; treat them as recovery evidence rather than a single clean
benchmark.

Provenance records platform, Python and NumPy versions, command, cwd, config and
harness hashes, containing Git commit/status, a source tree hash and thread
environment. The target Python environment inventory, installed Rust executable,
wheel hash, lock hash, CPU and physical memory are recorded when available.
On Windows the runner reads physical memory through
[GlobalMemoryStatusEx](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex).
`--mode definitive` requires `--wheel`, `--lock`, an accessible target Python,
CPU/memory, and installed Python sources and a Rust executable matching the wheel.
The CLI interpreter must resolve to the same Python environment and distribution
inventory supplied by `--environment-python`. Direct Python commands and scripts
with Python shebangs are supported; native launchers and shell wrappers fail
closed because their Python environment cannot be determined from the command.
The lock is an exhaustive plain text list of `package==version` entries, including
`lplearn` and all installed distributions; their normalized names and exact versions
must match the environment. Other requirements syntax fails closed. Supply at
least five repetitions, a repetition count divisible by the number of cases
for balanced ordering, and the LP/CVXPY case plus a matched Numba/Rust GDA or
SGDA pair.
It also rejects a run if the wheel, executable, interpreter, dependency lock,
installed package inventory, or source tree changes while measurements are running.
Qualification mode records absent evidence as null. These checks do not prove
that the lock was used to install packages or verify the bytes of every third-party
dependency. A prewarmed Numba cache must remain unchanged throughout a definitive
run. Run a small qualification first and review its raw receipts before any long job.

The CLI's Rust worker timeout is set uniformly with `--rust-timeout`, default
240 seconds; `--timeout` bounds each entire CLI invocation and defaults to 300
seconds. The outer timeout must be larger. Select limits before the run rather
than changing them after observing results.

Before a definitive run, record the workload matrix, cache mode, executable and
wheel hashes, lock, interpreter, machine, thread settings, free disk space,
dependencies and planned stopping conditions. Complete a disposable end-to-end
qualification with the same environment, including interruption, resume, and
terminal reload. Run the measurements on a machine without concurrent benchmark
or validation work; record material interference instead of treating such a
run as a clean performance result. Preserve raw shards and manifests under the
external `DATABASE_DIR`. The lock comparison checks installed names and versions;
it does not establish a reproducible build or verify all third-party package bytes.

## Provenance

This harness was prepared from `integration/lplearn-benchmark/benchmark.py` in
the private LP Learner integration workspace, building on the public Python and
Rust code whose attribution is recorded in [NOTICE](../NOTICE). The scope of this
PR is the benchmark protocol and its runner, not the manuscript's numerical
claims. No performance ratios are asserted by this repository without a verified
receipt-backed definitive run.
