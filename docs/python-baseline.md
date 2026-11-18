# Python baseline for queue policies

Install the local wheel as described in [the bridge guide](python-rust-bridge.md).
The package supports Python 3.12 and uses CVXPY/CLARABEL for the direct LP and
Numba or Rust for GDA/SGDA. Each command reads a single TOML configuration:

On Intel macOS, package metadata selects the last Numba release line with
official Intel macOS support and caps NumPy below 2.4, matching that Numba
line's tested range. Other platforms use the current Numba and NumPy ranges.
On Windows ARM64, CVXPY is kept below 1.9 until its `sparsediffpy` dependency
can be installed there without building OpenBLAS from source.

For example, save this as `policy.toml`:

```toml
[model]
K = 2
T = 3
case = "CJS"

[learning]
n_iter = 100
samples_per_pair = 10
seed = 0
```

The repository also includes `configs/policy.toml` with all learning options.

```sh
lplearn solve --config policy.toml --method lp --backend cvxpy --output results/lp.npz
lplearn solve --config policy.toml --method sgda --backend numba --output results/sgda.npz
lplearn solve --config policy.toml --method sgda --backend rust --output results/sgda-rust.npz
```

The `.npz` contains `occupation`, `config_json`, and `metadata_json` entries.
The occupation array is ordered `[time, state, action]`, with action
`u = 2*a + b`, where `a` is admission and `b` is the high service rate. It is
an occupation measure, not a conditional action probability. The printed output
path is absolute.

To solve the same configuration with CVXPY and draw its occupation measures:

```sh
lplearn plot-policy --config policy.toml --method lp --backend cvxpy --output results/lp.png
```

The default CVXPY solver is CLARABEL. To explicitly select another installed
CVXPY solver for this direct LP, pass `--cvxpy-solver`; for example:

```sh
lplearn plot-policy --config policy.toml --method lp --backend cvxpy --cvxpy-solver SCIPY --output results/lp-scipy.png
```

The selected solver is recorded in the JSON manifest. Different solvers can
return different optimal occupation measures and conditional policies for a
degenerate LP, even when their objective values and feasibility agree. Treat
these as solver-specific results; do not assume their policy plots are
identical. `--recover` currently uses CLARABEL for its projection and therefore
rejects a non-default `--cvxpy-solver` selection.

The short SGDA run in this example can return an infeasible iterate. To inspect
it as a diagnostic with either backend, explicitly allow that output:

```sh
lplearn plot-policy --config policy.toml --method sgda --backend numba --allow-infeasible --output results/numba-diagnostic.png
lplearn plot-policy --config policy.toml --method sgda --backend rust --allow-infeasible --output results/rust-diagnostic.svg --format svg
```

These SGDA plots show optimization iterates; they do not certify a feasible
policy. Check the recorded residuals before drawing conclusions from them.

Supported plots are PNG, PDF and SVG. Without `--output`, the command writes
`results/<config>-<method>-<backend>.png` (or the selected format). It prints
the absolute path, with a clickable terminal link when supported; `--open`
requests a viewer. Beside the figure, a `.npy` file contains the unmodified
occupation tensor and a `.json` manifest records the resolved configuration,
versions, timings, and SHA-256 hashes. Pie slices show occupation mass per
action; pie area represents state mass at that time. The visualization omits
negative entries for display only. It does not change the saved array or
discard small positive masses.

Each plot uses a new output stem. Existing plots, arrays, manifests and
recovery sidecars are never overwritten; choose a different `--output` path
when any of them already exists. Artifacts are prepared in a temporary
directory and published using same-filesystem hard links with the manifest
last. The filesystem must support hard links. Stems ending in `.policy` or
`.recovered` are reserved for sidecars. An interrupted process may leave a
lock or partial files without a manifest; inspect those before reusing the stem.
The lock coordinates `lplearn` writers; do not modify its output files from
another process during publication.

Before plotting, the CLI checks nonnegativity, initial condition, flow,
stage mass and resource constraints of the representative-arm LP relaxation
using an absolute tolerance of `1e-6`. Infeasible iterates fail by default.
To inspect one as a diagnostic, explicitly pass `--allow-infeasible`; the
figure and CLI identify it as an infeasible iterate and the JSON manifest
records each residual. Feasibility of the LP relaxation alone does not certify
sample-path constraints for a finite population.

For an explicit feasible recovery, use `--recover` instead:

```sh
lplearn plot-policy --config policy.toml --method sgda --backend rust --recover --output results/recovered.png
```

This numerically solves a CVXPY/CLARABEL quadratic program to find a closest
occupation measure in Euclidean distance satisfying the representative-arm LP flow,
nonnegativity and expected-resource constraints. It changes the learner's
iterate; it is not evidence that the learner converged or that the resulting
policy is optimal. The extra solve is included in the end-to-end command time.
The original iterate remains in `recovered.npy`; `recovered.recovered.npy`
contains the projected occupation, and `recovered.policy.npy` contains the
induced conditional policy. The manifest records both feasibility reports,
the projection distance and the feasibility of the induced policy evaluated
in the model. Feasibility is checked to a tolerance; optimality of the
projection distance is not independently certified, particularly if the
solver reports `optimal_inaccurate`. This validates the LP relaxation, not finite-population
sample-path resource limits. `--recover` and `--allow-infeasible` are mutually
exclusive.

The same path is available from Python:

```python
import lplearn as lpl

config = lpl.load_config("policy.toml")
occupation, metadata = lpl.solve(config, method="sgda", backend="numba")
```

`[model]` parameters and defaults are those of `lplearn.core.QueueParams`.
`[learning]` accepts `Gamma_y`, `Gamma_lambda`, `Gamma_mu`, `eta_y`,
`eta_dual`, `n_iter`, `samples_per_pair`, `seed`, `avg_window`, `record_every`,
and `init`. GDA forces zero samples; SGDA uses the positive sample count in
the configuration. The Numba solver's reported time includes any initial JIT
compilation. The manifest also gives model and total solver wall times.
For Rust, `--rust-timeout SECONDS` (or `solve(..., rust_timeout=SECONDS)`)
sets the worker timeout, default 30 seconds. The manifest separates worker
execution time, including model and solver setup, from the process/JSON time.
It records the Rust backend version and whether debug assertions were enabled.
The Numba and Rust internal timing boundaries differ, so compare full CLI wall
time or explicitly account for those differences. Rust errors are reported;
there is no fallback to Numba.

The copied numerical modules are attributed in [NOTICE](../NOTICE). This is a
single-job API; the older experiment runner's multi-job TOML grids are not
accepted as single-job input.
