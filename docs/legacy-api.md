# Legacy Python API compatibility

The `lplearn` namespace exposes the legacy top-level exports of
`lp-learner-new` at commit `c23e6007400daae28737f91d19205233d313d043`.
Existing code using those exports can change its import to:

```python
import lplearn as lpl

problem = lpl.Model(K=2, T=3, case=1)
result = problem.solve("cvxpy", verbose=False)
fig, ax = lpl.policy_pie_plot(result["solutions"]["cvxpy"]["y"].copy())
```

The legacy plotting function expects an occupation tensor shaped
`(K + 1, 2, 2, T)` and can modify its argument; copy the tensor when it must
be retained. The newer `load_config`, `solve`, and CLI interfaces remain
available. The compatibility code is loaded only when a legacy export is used.
The legacy `Model` implementation now initializes its `utils` dependency on
import so `solve` works after a fresh `import lplearn`.

Before clipping or normalizing the legacy plot, the function rejects
non-finite values and warns on values below `-1e-5` and stage mass that differs from
one by more than `1e-5`. Passing those checks does not verify the model's flow or resource
constraints; the function has no model from which to verify them. It retains
the historical plotting and input-mutation behavior.

The old optional NumPy/PyTorch comparison needs `pip install 'lplearn[torch]'`.
The experiment runner and its stages are outside this compatibility change.
The old distribution/import name `lp_learner` is not installed by this wheel;
update imports to `lplearn`. Source and license attribution are in NOTICE.
