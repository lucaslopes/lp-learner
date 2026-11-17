# Python baseline for queue policies

Install the local wheel as described in [the bridge guide](python-rust-bridge.md).
The package supports Python 3.12 and uses CVXPY/CLARABEL for the direct LP and
Numba for GDA/SGDA. Each command reads a single TOML configuration:

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
```

The `.npz` contains `occupation`, `config_json`, and `metadata_json` entries.
The occupation array is ordered `[time, state, action]`, with action
`u = 2*a + b`, where `a` is admission and `b` is the high service rate. It is
an occupation measure, not a conditional action probability. The printed output
path is absolute. A plot command and a Rust learning backend are separate work.

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

The copied numerical modules are attributed in [NOTICE](../NOTICE). This is a
single-job API; the older experiment runner's multi-job TOML grids are not
accepted as single-job input.
