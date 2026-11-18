"""Plot a queue occupation measure and record the input and output provenance.

The pie layout and colors are adapted from the LP Learner Python reference;
see NOTICE. The plotted data are occupation measures, not conditional policies.
"""

import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import webbrowser

import numpy as np

from .workflow import load_config, recover_occupation, solve


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_artifacts(staged, candidates):
    """Lock the output stem and publish the manifest last without overwrites."""
    output = staged[0][1]
    lock = output.with_name(f".{output.stem}.lplearn.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    os.close(fd)
    published = []
    try:
        for candidate in candidates:
            if os.path.lexists(candidate):
                raise FileExistsError(f"output artifact already exists: {candidate}")
        for source, target in staged:
            # Staging is in the destination directory. A hard link publishes
            # each file atomically and fails if another writer created it.
            os.link(source, target)
            published.append((source, target))
    except BaseException:
        for source, target in published:
            try:
                if os.path.samefile(source, target):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        lock.unlink(missing_ok=True)


def policy_pie_plot(occupation, *, feasibility=None, allow_infeasible=False):
    """Draw y[t,s,u], using u=2*a+b; do not mutate the caller's array.

    When a model-level feasibility report is omitted, only nonnegativity and
    stage mass can be checked; the plot is labeled as not model-verified.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch

    y = np.asarray(occupation, dtype=float)
    if y.ndim != 3 or y.shape[2] != 4 or not np.isfinite(y).all():
        raise ValueError("occupation must have finite shape (time, state, 4)")
    T, S, _ = y.shape
    if not T or not S:
        raise ValueError("occupation must contain at least one time and state")
    if feasibility is None:
        feasible = bool(y.min() >= -1e-6 and np.max(np.abs(y.sum(axis=(1, 2)) - 1.0)) <= 1e-6)
        title = "Occupation measures — model constraints not checked\nNegative entries omitted"
    else:
        feasible = feasibility["feasible"]
        title = "Occupation measures — feasible for the LP relaxation\nNegative entries omitted"
    if not feasible:
        if not allow_infeasible:
            raise ValueError("infeasible occupation iterate; use --allow-infeasible for a diagnostic plot")
        title = "INFEASIBLE ITERATE — diagnostic only\nNegative entries omitted"
    y = np.maximum(y, 0.0)
    colors = ("#f9a73e", "#bf212f", "#27b376", "#006f3c")
    fig, ax = plt.subplots(figsize=(max(4, T * 1.2), max(3, S * 1.2)))
    for t in range(T):
        masses = y[t].sum(axis=1)
        total = masses.sum()
        if total <= 0:
            continue
        # Pie area is proportional to state mass, as in the reference plot.
        fractions = masses / total
        for state, fraction in enumerate(fractions):
            if fraction > 0:
                ax.pie(y[t, state], radius=0.7 * np.sqrt(fraction),
                       center=(t, state), startangle=90, colors=colors)
    ax.set(xlim=(-0.8, T - 0.2), ylim=(-0.8, S - 0.2),
           xticks=range(T), yticks=range(S), xlabel="Arrival time (t)", ylabel="Queue size (s)")
    ax.set_title(title)
    ax.legend(handles=[Patch(color=c, label=f"a={u // 2}, b={u % 2}")
                       for u, c in enumerate(colors)], loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=4)
    return fig, ax


def plot_command(args):
    start = perf_counter()
    recover = getattr(args, "recover", False)
    config_path = Path(args.config).expanduser().resolve()
    initial_config_hash = _sha256(config_path)
    fmt = args.format or (Path(args.output).suffix.lstrip(".").lower() if args.output else "png")
    if fmt not in {"png", "pdf", "svg"}:
        raise ValueError("output extension or --format must be png, pdf, or svg")
    output = Path(args.output or f"results/{config_path.stem}-{args.method}-{args.backend}.{fmt}").expanduser().resolve()
    if output.suffix.lower() != f".{fmt}":
        raise ValueError("output extension must match --format")
    if output.stem.casefold().endswith((".recovered", ".policy")):
        raise ValueError("output stems ending in .recovered or .policy are reserved for sidecars")
    array_path, manifest_path = output.with_suffix(".npy"), output.with_suffix(".json")
    recovered_path = output.with_name(output.stem + ".recovered.npy")
    policy_path = output.with_name(output.stem + ".policy.npy")
    if config_path in (output, array_path, manifest_path, recovered_path, policy_path):
        raise ValueError("output artifacts must not overwrite the configuration")
    candidates = (output, array_path, manifest_path, recovered_path, policy_path)
    for candidate in candidates:
        if os.path.lexists(candidate):
            raise FileExistsError(f"output artifact already exists: {candidate}; choose a new --output path")
    config = load_config(config_path)
    occupation, timing = solve(config, args.method, args.backend, rust_timeout=args.rust_timeout)
    if _sha256(config_path) != initial_config_hash:
        raise ValueError("configuration changed during the solve")
    feasibility = timing["occupation_feasibility"]
    plotted_occupation = occupation
    if recover:
        from .core import QueueParams, queue_wcmdp
        from .core.policy import conditional_policy, fluid_evaluation
        fields = dict(config["model"])
        if fields.get("m0") is not None:
            fields["m0"] = tuple(fields["m0"])
        model = queue_wcmdp(QueueParams(**fields))
        plotted_occupation, recovery = recover_occupation(occupation, model)
        timing["recovery"] = recovery
        feasibility = recovery["recovered_feasibility"]
        policy = conditional_policy(plotted_occupation)
        from .workflow import occupation_feasibility
        policy_check = occupation_feasibility(fluid_evaluation(model, policy)["y"], model)
        if not policy_check["feasible"]:
            raise RuntimeError(f"recovered conditional policy failed LP feasibility: {policy_check}")
        recovery["conditional_policy_feasibility"] = policy_check
    if not feasibility["feasible"] and not args.allow_infeasible:
        max_violation = max(feasibility[name] for name in (
            "nonnegativity_violation", "initial_residual", "flow_residual",
            "stage_mass_residual", "resource_violation"))
        raise ValueError("infeasible occupation iterate (max nonnegativity/initial/flow/mass/resource violation "
                         f"{max_violation:.6g}); "
                         "use --allow-infeasible for a diagnostic plot")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lplearn-", dir=output.parent) as temp:
        stage = Path(temp)
        plot_stage, array_stage = stage / output.name, stage / array_path.name
        recovered_stage, policy_stage = stage / recovered_path.name, stage / policy_path.name
        manifest_stage = stage / manifest_path.name
        plot_start = perf_counter()
        fig, ax = policy_pie_plot(plotted_occupation, feasibility=feasibility, allow_infeasible=args.allow_infeasible)
        if recover:
            ax.set_title("Recovered occupation (LP projection)\nNegative entries omitted")
        from matplotlib import pyplot as plt
        try:
            fig.savefig(plot_stage, format=fmt, bbox_inches="tight")
        finally:
            plt.close(fig)
        timing["plot_seconds"] = perf_counter() - plot_start
        np.save(array_stage, occupation)
        if recover:
            np.save(recovered_stage, plotted_occupation)
            np.save(policy_stage, policy)
            timing["recovery"].update(
                occupation_path=str(recovered_path), occupation_sha256=_sha256(recovered_stage),
                conditional_policy_path=str(policy_path), conditional_policy_sha256=_sha256(policy_stage),
            )
        timing.update(config=config, config_path=str(config_path), config_sha256=initial_config_hash,
                      plot_kind=("recovered_occupation" if recover else
                                 "occupation" if feasibility["feasible"] else "infeasible_iterate_diagnostic"),
                      display_threshold=0.0,
                      plot_path=str(output), plot_sha256=_sha256(plot_stage),
                      occupation_path=str(array_path), occupation_sha256=_sha256(array_stage),
                      python=sys.version.split()[0],
                      dependency_versions={name: package_metadata.version(name) for name in
                                           ("lplearn", "numpy", "scipy", "cvxpy", "numba", "matplotlib")},
                      in_process_seconds=perf_counter() - start)
        manifest_stage.write_text(json.dumps(timing, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged = [(plot_stage, output), (array_stage, array_path)]
        if recover:
            staged.extend(((recovered_stage, recovered_path), (policy_stage, policy_path)))
        staged.append((manifest_stage, manifest_path))
        _publish_artifacts(staged, candidates)
    print(f"Policy plot: {output}")
    if recover:
        print("lplearn: plotted an LP-feasible projection; the original iterate is saved separately", file=sys.stderr)
    if not feasibility["feasible"]:
        print("lplearn: infeasible iterate plotted for diagnostics only; negative entries are omitted", file=sys.stderr)
    if sys.stdout.isatty():
        print(f"\033]8;;{output.as_uri()}\033\\Open plot\033]8;;\033\\")
    if args.open and not webbrowser.open(output.as_uri()):
        print("lplearn: could not open plot in a viewer", file=sys.stderr)
        return 1
    return 0
