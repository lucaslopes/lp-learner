'''
Publication figures for the SIGMETRICS manuscript (``papers/SIGMETRICS/main.tex``), drawn from recorded results.

Each figure has a plotting function that takes explicit inputs (occupation measures, a distance matrix, or
DataFrames), returns a matplotlib ``Figure`` and, when ``out_path`` is given, saves it as a vector PDF with
embedded TrueType fonts. ``FigureContext`` reads the recorded results of the experiment stages from a results root,
checks their schema and records the files each figure used; ``build_figures`` draws the requested figures. The
``figures`` stage (``lp_learner/experiments/stages/figures.py``) and the command line both call ``build_figures``::

    uv run python -m lplearn.visualizations.paper_figures --results-root results \\
        --out-dir papers/SIGMETRICS/figures/generated [--only NAME ...] [--config configs/figures.toml]

Conventions
-----------
- Occupation measures use the core layout ``y[t, s, u]`` with ``u = 2a + b`` (shape ``(T, S, 4)``). The legacy
  helper ``policy_pie_plot`` expects ``y[s, a, b, t]``; ``to_legacy_layout`` converts, returning a copy because the
  helper clips its input in place.
- Pies: at each stage ``t`` the state masses are computed after clipping entries below 0.01 to zero, rounded to two
  decimals and normalised over the states, so every stage sums to one and leaked mass is removed. The area of the
  pie at ``(t, s)`` is proportional to that share; its wedges are the split over the four actions (a, b).
- Distances are Frobenius norms between LP-level occupation measures. They are structural diagnostics,
  not finite-population performance measures.
'''

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import colormaps, rc_context
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from lplearn._legacy.utils import palettes
from lplearn._legacy.visualizations.policy import policy_pie_plot


DEFAULT_RESULTS_ROOT = 'results'
DEFAULT_OUT_DIR = 'papers/SIGMETRICS/figures/generated'
FONT_FAMILY = 'Times New Roman'
CREATOR = 'lp_learner.visualizations.paper_figures'
CASES = ('CJS', 'SJS')
INSTANCE_SHAPES = {'K5T10': (10, 6), 'K11T30': (30, 12)}  # (T, S) of each instance, S = K + 1
EXACT_LABELS = ('A', 'B')  # interior-point LP solutions of the lp_solutions stage; every other label is learned
EXACT_KINDS = {'A': 'lp', 'B': 'lp_reg'}  # solve kind of each exact label in the lp_solutions records


def _exact_kind_partition(label: str) -> tuple[str, str]:
    '''Solve kind and partition of an exact label: A, B on the detailed grid, or '<partition>-LP' (aggregate LP, lifted).'''
    if label in EXACT_KINDS:
        return EXACT_KINDS[label], NO_PARTITION
    if label.endswith('-LP'):
        return 'lp', label[:-3]
    raise KeyError(f'not an exact label: {label!r}')
NO_PARTITION = 'none'  # partition label of the detailed grid in the records
PIE_MIN_VALUE = 1e-2
PIE_COLORS = (palettes.DEFAULT['a0b0'], palettes.DEFAULT['a0b1'], palettes.DEFAULT['a1b0'], palettes.DEFAULT['a1b1'])
PIE_LABELS = (r'$y_{s,0,0}(t)$', r'$y_{s,0,1}(t)$', r'$y_{s,1,0}(t)$', r'$y_{s,1,1}(t)$')

# Method colours: Okabe-Ito for A-G and Paul Tol's muted colours for H-K (colourblind-safe, one colour per label).
METHOD_COLORS = {
    'A': '#000000', 'B': '#0072B2', 'C': '#E69F00', 'D': '#009E73', 'E': '#56B4E9', 'F': '#D55E00', 'G': '#CC79A7',
    'H': '#44AA99', 'I': '#882255', 'J': '#332288', 'K': '#999933',
}
METHOD_MARKERS = dict(zip('ABCDEFGHIJK', ('o', 's', '^', 'D', 'v', 'P', 'X', 'h', '<', '>', '*')))
UPDATE_LABELS = {'projection': 'Projection', 'lp_update': 'LP-update'}
POPULATION_LABELS = {'coord': 'Coordinated', 'indep': 'Independent', 'selfish': 'Selfish', 'price': 'Price-taking'}
POPULATION_COLORS = {'coord': '#0072B2', 'indep': '#D55E00', 'selfish': '#009E73', 'price': '#E69F00'}
MECHANISM_LABELS = {'M0': 'M0 unenforced', 'M1': 'M1 rationing', 'M2': 'M2 congestion'}
CONVENTION_STYLE = {'paper': ('#0072B2', 'o'), 'consistent': ('#D55E00', 's')}
CONVENTION_LABELS = {'paper': 'paper', 'consistent': 'consistent'}
RESOURCES = (('admission', 'Admission ($\\ell=0$)'), ('high_rate', 'High rate ($\\ell=1$)'))

# Serif text everywhere (the manuscript body is Times), embedded TrueType fonts, mathtext matched to Times.
PAPER_RC = {
    'font.family': 'serif',
    'font.serif': [FONT_FAMILY, 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'lines.linewidth': 1.1,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'legend.borderpad': 0.4,
}

# Columns that the plotting functions need (the loaders map the stages' files onto these where they differ).
PLOT_SCHEMAS: dict[str, tuple[str, ...]] = {
    'finite_n': ('case', 'method', 'policy', 'N', 'cost_mean', 'cost_ci_lo', 'cost_ci_hi', 'lp_objective'),
    'bound': ('case', 'partition', 'convention', 'regime', 'true_error', 'bound'),
    'commons_costs': ('mechanism', 'population', 'cost_mean', 'cost_ci_lo', 'cost_ci_hi'),
    'commons_stages': ('population', 'stage', 'metric', 'mean', 'ci_low', 'ci_high'),
    'dual': ('case', 'method', 'resource', 't', 'mean', 'sd'),
}
# Columns that the stages write (checked by the loaders). `commons_stages` is requested from the commons stage.
FILE_SCHEMAS: dict[str, tuple[str, ...]] = {
    'finite_n_summary': ('case', 'method', 'policy', 'N', 'cost_mean', 'cost_ci_lo', 'cost_ci_hi', 'lp_objective'),
    'bound_rows': ('K', 'T', 'case', 'label', 'convention', 'regime', 'dist_total'),
    'commons_summary': ('scenario', 'case', 'N', 'mechanism', 'tie_break', 'population', 'cost_mean', 'cost_ci95'),
    'commons_stages': ('case', 'N', 'mechanism', 'tie_break', 'population', 'stage', 'metric', 'mean', 'ci_low', 'ci_high'),
}


class FigureInputError(RuntimeError):
    '''
    An input of a figure is missing, or does not have the documented schema.
    '''


# ---------------------------------------------------------------------------
# Occupation-measure layouts and the pie panels
# ---------------------------------------------------------------------------


def to_legacy_layout(y: np.ndarray) -> np.ndarray:
    '''
    Core layout y[t, s, u] (u = 2a + b) to the legacy layout y[s, a, b, t] of ``policy_pie_plot``. Returns a new array.
    '''

    y = np.asarray(y, dtype=float)
    if y.ndim != 3 or y.shape[2] != 4:
        raise ValueError(f'expected an occupation measure of shape (T, S, 4), got {y.shape}')
    T, S, _ = y.shape
    return np.array(y.reshape(T, S, 2, 2).transpose(1, 2, 3, 0), dtype=float, order='C', copy=True)


def from_legacy_layout(y_legacy: np.ndarray) -> np.ndarray:
    '''
    Inverse of ``to_legacy_layout``: legacy y[s, a, b, t] to core y[t, s, u] with u = 2a + b. Returns a new array.
    '''

    y = np.asarray(y_legacy, dtype=float)
    if y.ndim != 4 or y.shape[1:3] != (2, 2):
        raise ValueError(f'expected a legacy occupation measure of shape (S, 2, 2, T), got {y.shape}')
    S, _, _, T = y.shape
    return np.array(y.transpose(3, 0, 1, 2).reshape(T, S, 4), dtype=float, order='C', copy=True)


@dataclass(frozen=True, eq=False)
class PolicyPanel:
    '''
    One pie-grid panel: a title and an occupation measure y[t, s, u]; `source` names the recorded arrays it came from.
    '''

    title: str
    y: np.ndarray
    source: str = ''


def draw_policy_pies(ax: Any, y: np.ndarray, *, pie_fontsize: float, label_fontsize: float,
                     min_radius: float = 0.1, max_radius: float = 0.7) -> None:
    '''
    Draw the stage-wise pies of `y` on `ax` with the legacy helper (same geometry, palette and thresholds).
    '''

    legacy = to_legacy_layout(y)  # a copy: policy_pie_plot clips its input in place
    # A stage without mass gives 0/0 in the helper; it then draws nothing for that stage, so the warning is silenced.
    with np.errstate(divide='ignore', invalid='ignore'):
        policy_pie_plot(
            legacy,
            min_radius=min_radius,
            max_radius=max_radius,
            fontsize=pie_fontsize,
            label_fontsize=label_fontsize,
            fontfamily=FONT_FAMILY,
            color_palette=dict(palettes.DEFAULT),
            min_value=PIE_MIN_VALUE,
            show_legend=False,
            ax=ax,
        )


def _pie_legend_handles() -> list[Patch]:
    return [Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=label) for color, label in zip(PIE_COLORS, PIE_LABELS)]


def _pie_grid(rows: Sequence[Sequence[PolicyPanel]], *, figsize: tuple[float, float], pie_fontsize: float,
              label_fontsize: float, title_fontsize: float, legend_fontsize: float, tick_step: int = 1,
              out_path: str | Path | None) -> Figure:
    if not rows or any(not row for row in rows):
        raise FigureInputError('a policy grid needs at least one panel in every row')
    n_cols = max(len(row) for row in rows)
    has_invalid = any(
        np.min(panel.y) < -1e-5 or np.max(np.abs(np.sum(panel.y, axis=(1, 2)) - 1.0)) > 1e-5
        for row in rows for panel in row
    )
    # The extra diagnostic title line needs vertical space at publication size.
    if has_invalid:
        figsize = (figsize[0], figsize[1] + 0.8 * len(rows))
    with rc_context(PAPER_RC):
        fig = Figure(figsize=figsize, layout='constrained')
        axes = fig.subplots(len(rows), n_cols, squeeze=False)
        for r, row in enumerate(rows):
            for c in range(n_cols):
                ax = axes[r, c]
                if c >= len(row):
                    ax.set_axis_off()
                    continue
                panel = row[c]
                draw_policy_pies(ax, panel.y, pie_fontsize=pie_fontsize, label_fontsize=label_fontsize)
                if tick_step > 1:  # long horizons: label every `tick_step`-th stage only
                    ax.set_xticks(list(range(0, panel.y.shape[0], tick_step)))
                values = np.asarray(panel.y, dtype=float)
                invalid = bool(values.min() < -1e-5 or
                               np.max(np.abs(values.sum(axis=(1, 2)) - 1.0)) > 1e-5)
                title = panel.title + ('\nInfeasible iterate (clipped)' if invalid else '')
                ax.set_title(title, fontsize=title_fontsize, pad=3)
        fig.legend(handles=_pie_legend_handles(), loc='outside upper center', ncol=4, fontsize=legend_fontsize)
        if out_path is not None:
            save_figure(fig, out_path)
    return fig


def _check_panels(panels: Sequence[PolicyPanel], name: str) -> list[PolicyPanel]:
    panels = list(panels)
    if not panels:
        raise FigureInputError(f'{name}: no policy panels given')
    for panel in panels:
        y = np.asarray(panel.y, dtype=float)
        if y.ndim != 3 or y.shape[2] != 4 or not np.all(np.isfinite(y)):
            raise FigureInputError(f'{name}: panel {panel.title!r} is not a finite (T, S, 4) occupation measure: {y.shape}')
    return panels


def plot_policy_solutions_non_agg(cjs: Sequence[PolicyPanel], sjs: Sequence[PolicyPanel],
                                  out_path: str | Path | None = None, *, figsize: tuple[float, float] = (7.0, 3.5)) -> Figure:
    '''
    Figure (a): detailed (non-aggregated) policies, one row per case (CJS, SJS), one column per method.
    '''

    rows = [_check_panels(cjs, 'policy_solutions_non_agg (CJS)'), _check_panels(sjs, 'policy_solutions_non_agg (SJS)')]
    return _pie_grid(rows, figsize=figsize, pie_fontsize=9.5, label_fontsize=11.5, title_fontsize=10.5, legend_fontsize=10.5,
                     out_path=out_path)


def plot_policy_solutions_agg(cjs: Sequence[PolicyPanel], sjs: Sequence[PolicyPanel],
                              out_path: str | Path | None = None, *, figsize: tuple[float, float] = (7.0, 3.5)) -> Figure:
    '''
    Figure (b): aggregate-method policies (lifted to the detailed grid), one row per case, one column per partition.
    '''

    rows = [_check_panels(cjs, 'policy_solutions_agg (CJS)'), _check_panels(sjs, 'policy_solutions_agg (SJS)')]
    return _pie_grid(rows, figsize=figsize, pie_fontsize=9.5, label_fontsize=11.5, title_fontsize=10.5, legend_fontsize=10.5,
                     out_path=out_path)


def plot_policies_K11T30(panels: Sequence[PolicyPanel], out_path: str | Path | None = None, *,
                         figsize: tuple[float, float] = (7.0, 4.6)) -> Figure:
    '''
    Figure (d): the K = 11, T = 30 CJS instance in a 2 x 2 grid (row-major order of `panels`).
    '''

    panels = _check_panels(panels, 'policies_K11T30')
    if len(panels) != 4:
        raise FigureInputError(f'policies_K11T30 needs exactly four panels, got {len(panels)}')
    rows = [panels[:2], panels[2:]]
    return _pie_grid(rows, figsize=figsize, pie_fontsize=8.5, label_fontsize=10.5, title_fontsize=10, legend_fontsize=10,
                     tick_step=2, out_path=out_path)


# ---------------------------------------------------------------------------
# Distance heatmap
# ---------------------------------------------------------------------------


def _annotation_color(rgba: Sequence[float]) -> str:
    '''
    Dark text on light cells and white on dark cells, using the relative-luminance rule of seaborn.
    '''

    rgb = np.asarray(rgba[:3], dtype=float)
    linear = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
    return '.15' if luminance > 0.408 else 'white'


def plot_heatmap_policies_compare(labels: Sequence[str], cjs: np.ndarray, sjs: np.ndarray,
                                  out_path: str | Path | None = None, *, value_range: tuple[float, float] = (0.0, 3.0),
                                  figsize: tuple[float, float] = (7.0, 3.5)) -> Figure:
    '''
    Figure (c): pairwise Frobenius distances among the methods, CJS and SJS side by side with one shared colour bar.
    `cjs` and `sjs` are square matrices indexed like `labels`.
    '''

    labels = [str(label) for label in labels]
    n = len(labels)
    matrices = []
    for name, matrix in (('CJS', cjs), ('SJS', sjs)):
        mat = np.asarray(matrix, dtype=float)
        if mat.shape != (n, n) or not np.all(np.isfinite(mat)):
            raise FigureInputError(f'heatmap_policies_compare: {name} matrix must be finite and of shape ({n}, {n}), got {mat.shape}')
        matrices.append(mat)
    vmin, vmax = float(value_range[0]), float(value_range[1])
    cmap = colormaps['RdYlGn_r']
    norm = Normalize(vmin=vmin, vmax=vmax)
    with rc_context(PAPER_RC):
        fig = Figure(figsize=figsize, layout='constrained')
        grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.045])
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
        cax = fig.add_subplot(grid[0, 2])
        mesh = None
        for ax, mat, title in zip(axes, matrices, ('CJS', 'SJS')):
            mesh = ax.pcolormesh(mat, cmap=cmap, norm=norm)
            ax.set_xlim(0, n)
            ax.set_ylim(n, 0)
            ax.set_aspect('equal')
            for i in range(n):
                for j in range(n):
                    ax.text(j + 0.5, i + 0.5, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=6.6,
                            color=_annotation_color(cmap(norm(mat[i, j]))))
            ax.set_xticks(np.arange(n) + 0.5, [f'({label})' for label in labels], fontsize=8)
            ax.set_yticks(np.arange(n) + 0.5, [f'({label})' for label in labels], fontsize=8)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(title, fontsize=10, pad=3)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label('Frobenius norm of the difference between solutions', fontsize=8)
        cbar.ax.tick_params(labelsize=7.5, length=2)
        if out_path is not None:
            save_figure(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Finite-population costs, bound validation, dual prices and the no-coordinator experiment
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FigureInputError(f'{name}: missing columns {missing}; found {list(frame.columns)}')
    if frame.empty:
        raise FigureInputError(f'{name}: no rows to plot')
    return frame


def _require_intervals(frame: pd.DataFrame, name: str, *, mean: str, low: str, high: str) -> None:
    '''
    Every plotted interval must bracket its estimate: low <= estimate <= high (NaN fails).
    '''

    lo = frame[low].astype(float).to_numpy()
    mid = frame[mean].astype(float).to_numpy()
    hi = frame[high].astype(float).to_numpy()
    bad = ~((lo <= mid + 1e-12) & (mid <= hi + 1e-12))
    if bad.any():
        raise FigureInputError(f'{name}: {int(bad.sum())} row(s) whose {mean} lies outside [{low}, {high}]')


def _method_handles(methods: Sequence[str]) -> list[Line2D]:
    return [Line2D([], [], color=METHOD_COLORS.get(m, '0.4'), marker=METHOD_MARKERS.get(m, 'o'), ms=3.5, lw=1.1, label=m)
            for m in methods]


def plot_finite_n_costs(frame: pd.DataFrame, out_path: str | Path | None = None, *, methods: Sequence[str],
                        policies: Sequence[str] = ('projection', 'lp_update'), cases: Sequence[str] = CASES,
                        figsize: tuple[float, float] = (7.0, 2.5)) -> Figure:
    '''
    Figure (e): mean cost per queue with a 95% CI against the number of queues N (log scale). One column per case and
    one row per update rule (projection, LP-update). For each selected method, its exact-LP aggregate counterpart
    (label ``<method>-LP``) is drawn dash-dotted with hollow markers when present. The dashed black line is the LP value
    of method (A).
    `frame` has the columns of ``PLOT_SCHEMAS['finite_n']`` (the stage's ``finite_n_summary.csv``).
    '''

    frame = _require_columns(frame, PLOT_SCHEMAS['finite_n'], 'finite_n_costs')
    _require_intervals(frame, 'finite_n_costs', mean='cost_mean', low='cost_ci_lo', high='cost_ci_hi')
    available = set(frame['method'].astype(str))
    chosen = []
    for method in methods:
        for label in (method, f'{method}-LP'):
            if label in available:
                chosen.append(label)
    if not chosen:
        raise FigureInputError(f'finite_n_costs: none of the methods {list(methods)} (or their -LP counterparts) is in the summary')
    present = set(frame['policy'].astype(str))
    rows = [p for p in policies if p in present]  # one row per configured update rule that the summary has
    if not rows:
        raise FigureInputError(f'finite_n_costs: none of the policies {list(policies)} is in the summary')
    with rc_context(PAPER_RC):
        fig = Figure(figsize=(figsize[0], figsize[1] * len(rows) + 0.25), layout='constrained')
        axes = fig.subplots(len(rows), len(cases), squeeze=False, sharex=True)
        for r, policy in enumerate(rows):
            for c, case in enumerate(cases):
                ax = axes[r, c]
                sub = frame[(frame['case'] == case) & (frame['policy'] == policy)]
                ns = sorted(set(sub['N'].astype(int)))
                for label in chosen:
                    part = sub[sub['method'] == label].sort_values('N')
                    if part.empty:
                        continue
                    base = label[:-3] if label.endswith('-LP') else label
                    color = METHOD_COLORS.get(base, '0.4')
                    counterpart = label.endswith('-LP')
                    ax.plot(part['N'], part['cost_mean'], color=color, marker=METHOD_MARKERS.get(base, 'o'), ms=3.2, lw=1.1,
                            ls='-.' if counterpart else '-', mfc='white' if counterpart else color)
                    ax.fill_between(part['N'], part['cost_ci_lo'], part['cost_ci_hi'], color=color, alpha=0.14, lw=0)
                reference = _lp_reference(frame, case)
                ax.axhline(reference, color='black', ls='--', lw=0.9, zorder=3)
                ax.set_xscale('log')
                ax.set_xticks(ns, [str(n) for n in ns])
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(labelsize=8)
                ax.grid(axis='y', color='0.9', lw=0.6, zorder=-1)
                if r == 0:
                    ax.set_title(case, fontsize=10, pad=3)
                if r == len(rows) - 1:
                    ax.set_xlabel(r'Number of queues $N$', fontsize=9)
                if c == 0:
                    ax.set_ylabel(f'{UPDATE_LABELS.get(policy, policy)}\ncost per queue', fontsize=8.5)
        handles = [Line2D([], [], color=METHOD_COLORS.get(l[:-3] if l.endswith('-LP') else l, '0.4'), lw=1.1,
                          ls='-.' if l.endswith('-LP') else '-', marker=METHOD_MARKERS.get(l[:-3] if l.endswith('-LP') else l, 'o'),
                          ms=3.5, label=l) for l in chosen]
        handles.append(Line2D([], [], color='black', ls='--', lw=0.9, label='LP value of (A)'))
        fig.legend(handles=handles, loc='outside upper center', ncol=min(len(handles), 8), fontsize=8.0, handlelength=2.2)
        if out_path is not None:
            save_figure(fig, out_path)
    return fig


def _lp_reference(frame: pd.DataFrame, case: str) -> float:
    rows = frame[(frame['case'] == case) & (frame['method'].astype(str) == 'A')]['lp_objective'].dropna()
    if rows.empty:
        raise FigureInputError(f'finite_n_costs: no lp_objective for method (A) in case {case}')
    return float(rows.iloc[0])


def plot_bound_validation(frame: pd.DataFrame, out_path: str | Path | None = None, *, partitions: Sequence[str],
                          regimes: Sequence[tuple[str, str]], figsize: tuple[float, float] = (6.4, 5.4)) -> Figure:
    '''
    Figure (f): true error ||w* - w_bar*|| (horizontal) against the bound (vertical), both on log scales, for each
    partition, lifting convention and regime (rows: `regimes` as (name, label) pairs; columns: cases). Points above
    the shaded diagonal satisfy bound >= true error. `frame` has the columns of ``PLOT_SCHEMAS['bound']``; rows without
    a finite positive bound are counted, not plotted.
    '''

    frame = _require_columns(frame, PLOT_SCHEMAS['bound'], 'bound_validation')
    with rc_context(PAPER_RC):
        fig = Figure(figsize=figsize, layout='constrained')
        axes = fig.subplots(len(regimes), len(CASES), squeeze=False)
        for r, (regime, regime_label) in enumerate(regimes):
            for c, case in enumerate(CASES):
                ax = axes[r, c]
                sub = frame[(frame['case'] == case) & (frame['regime'].astype(str) == regime)
                            & (frame['partition'].astype(str).isin(partitions))]
                finite = sub[np.isfinite(sub['true_error']) & np.isfinite(sub['bound'])
                             & (sub['true_error'] > 0) & (sub['bound'] > 0)]  # log scales need positive values
                n_missing = int(len(sub) - len(finite))
                violations = int((finite['bound'] < finite['true_error']).sum())
                lo = float(min(finite['true_error'].min(), finite['bound'].min())) if len(finite) else 1e-6
                hi = float(max(finite['true_error'].max(), finite['bound'].max())) if len(finite) else 1.0
                lo, hi = lo / 1.6, hi * 1.6
                ax.fill_between([lo, hi], [lo, hi], [hi, hi], color='0.92', lw=0, zorder=0)
                ax.plot([lo, hi], [lo, hi], color='0.25', ls='--', lw=0.8, zorder=1)
                for convention, (color, marker) in CONVENTION_STYLE.items():
                    part = finite[finite['convention'] == convention]
                    ax.scatter(part['true_error'], part['bound'], s=16, marker=marker, facecolor='white',
                               edgecolor=color, linewidths=0.9, zorder=3)
                    # labels of the two conventions go to opposite sides, so clustered points stay readable
                    offset, align = ((-3, 3), 'right') if convention == 'paper' else ((3, -6), 'left')
                    for _, row in part.iterrows():
                        ax.annotate(str(row['partition']), (row['true_error'], row['bound']), xytext=offset,
                                    textcoords='offset points', fontsize=6.5, color=color, ha=align)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.tick_params(labelsize=7.5)
                # the check sits in the empty corner below the diagonal, away from the points
                checked = len(finite) - violations
                note = f'bound $\\geq$ true: {checked}/{len(finite)}' if len(finite) else 'no finite bound'
                if n_missing:
                    note += f'; no valid bound: {n_missing}'
                ax.text(0.97, 0.05, note, transform=ax.transAxes, fontsize=6.8, ha='right', color='0.25')
                if r == 0:
                    ax.set_title(case, fontsize=10, pad=3)
                if c == 0:
                    ax.set_ylabel(f'{regime_label}\nbound', fontsize=8.5)
                if r == len(regimes) - 1:
                    ax.set_xlabel(r'$\|w^\star-\bar w^\star\|$ (true)', fontsize=9)
        handles = [Line2D([], [], color=color, marker=marker, mfc='white', mec=color, ls='', ms=4.5, label=CONVENTION_LABELS[c])
                   for c, (color, marker) in CONVENTION_STYLE.items()]
        fig.legend(handles=handles, loc='outside upper center', ncol=2, fontsize=8.5, title='lifting convention', title_fontsize=8.5)
        if out_path is not None:
            save_figure(fig, out_path)
    return fig


def plot_dual_prices(frame: pd.DataFrame, out_path: str | Path | None = None, *, methods: Sequence[str],
                     cases: Sequence[str] = CASES, figsize: tuple[float, float] = (6.4, 4.8)) -> Figure:
    '''
    Figure (g): resource prices lambda_res[l, t] over the stages, mean and +/- one standard deviation over seeds.
    Rows are cases, columns are the admission (l = 0) and high-rate (l = 1) resources.
    `frame` has the columns of ``PLOT_SCHEMAS['dual']``.
    '''

    frame = _require_columns(frame, PLOT_SCHEMAS['dual'], 'dual_prices')
    methods = [m for m in methods if m in set(frame['method'].astype(str))]
    if not methods:
        raise FigureInputError('dual_prices: none of the selected methods appear in the data')
    with rc_context(PAPER_RC):
        fig = Figure(figsize=figsize, layout='constrained')
        axes = fig.subplots(len(cases), len(RESOURCES), squeeze=False, sharex=True)
        for r, case in enumerate(cases):
            for c, (resource, resource_label) in enumerate(RESOURCES):
                ax = axes[r, c]
                for method in methods:
                    part = frame[(frame['case'] == case) & (frame['method'].astype(str) == method) & (frame['resource'] == resource)]
                    part = part.sort_values('t')
                    if part.empty:
                        continue
                    color = METHOD_COLORS.get(method, '0.4')
                    sd = part['sd'].fillna(0.0)
                    ax.plot(part['t'], part['mean'], color=color, marker=METHOD_MARKERS.get(method, 'o'), ms=2.6, lw=1.0)
                    ax.fill_between(part['t'], part['mean'] - sd, part['mean'] + sd, color=color, alpha=0.15, lw=0)
                ax.grid(axis='y', color='0.9', lw=0.6, zorder=-1)
                ax.tick_params(labelsize=8)
                if r == 0:
                    ax.set_title(resource_label, fontsize=9.5, pad=3)
                if r == len(cases) - 1:
                    ax.set_xlabel(r'Stage $t$', fontsize=9)
                if c == 0:
                    ax.set_ylabel(f'{case}\nresource price', fontsize=8.5)
        fig.legend(handles=_method_handles(methods), loc='outside upper center', ncol=len(methods), fontsize=8.5, handlelength=2.2)
        if out_path is not None:
            save_figure(fig, out_path)
    return fig


def plot_commons_overload(stages: pd.DataFrame, costs: pd.DataFrame, out_path: str | Path | None = None, *,
                          populations: Sequence[str], mechanisms: Sequence[str],
                          figsize: tuple[float, float] = (7.0, 2.7)) -> Figure:
    '''
    Figure (h): the no-coordinator experiment. Left and middle: requests divided by budget, stage by stage, for
    admission and high rate, one line per population (the dotted line is the budget). Right: cost per queue with its
    95% CI for each mechanism, one bar per population. `stages` has the columns of ``PLOT_SCHEMAS['commons_stages']``
    and `costs` those of ``PLOT_SCHEMAS['commons_costs']``.
    '''

    stages = _require_columns(stages, PLOT_SCHEMAS['commons_stages'], 'commons_overload (stages)')
    costs = _require_columns(costs, PLOT_SCHEMAS['commons_costs'], 'commons_overload (costs)')
    _require_intervals(stages, 'commons_overload (stages)', mean='mean', low='ci_low', high='ci_high')
    _require_intervals(costs, 'commons_overload (costs)', mean='cost_mean', low='cost_ci_lo', high='cost_ci_hi')
    populations = [p for p in populations if p in set(stages['population'].astype(str))]
    if not populations:
        raise FigureInputError('commons_overload: none of the populations appear in the stage table')
    ratio_panels = (('ratio_admission', 'Admission: requests / budget'), ('ratio_high_rate', 'High rate: requests / budget'))
    with rc_context(PAPER_RC):
        fig = Figure(figsize=figsize, layout='constrained')
        axes = fig.subplots(1, 3)
        for ax, (metric, title) in zip(axes[:2], ratio_panels):
            sub = stages[stages['metric'] == metric]
            if sub.empty:
                raise FigureInputError(f'commons_overload: no stage rows for metric {metric!r}')
            for population in populations:
                part = sub[sub['population'].astype(str) == population].sort_values('stage')
                if part.empty:
                    continue
                color = POPULATION_COLORS.get(population, '0.4')
                ax.plot(part['stage'], part['mean'], color=color, lw=1.2)
                ax.fill_between(part['stage'], part['ci_low'], part['ci_high'], color=color, alpha=0.16, lw=0)
            ax.axhline(1.0, color='black', ls=':', lw=0.9)
            ax.set_title(title, fontsize=9.5, pad=3)
            ax.set_xlabel(r'Stage $t$', fontsize=9)
            ax.grid(axis='y', color='0.9', lw=0.6, zorder=-1)
            ax.tick_params(labelsize=8)
        # Right panel: cost relative to the coordinated policy under the enforcing mechanisms. Without enforcement
        # (M0) selfish queues are cheaper only because they exceed the budgets, so M0 is left to the table.
        cost_ax = axes[2]
        enforcing = [m for m in mechanisms if m != 'M0']
        others = [p for p in populations if p != 'coord']
        width = 0.8 / max(len(others), 1)
        for g, mechanism in enumerate(enforcing):
            ref = costs[(costs['mechanism'].astype(str) == mechanism) & (costs['population'].astype(str) == 'coord')]
            if ref.empty:
                continue
            ref = ref.iloc[0]
            ref_half = (ref['cost_ci_hi'] - ref['cost_ci_lo']) / 2
            for p, population in enumerate(others):
                part = costs[(costs['mechanism'].astype(str) == mechanism) & (costs['population'].astype(str) == population)]
                if part.empty:
                    continue
                row = part.iloc[0]
                x = g + (p - (len(others) - 1) / 2) * width
                rel = 100 * (row['cost_mean'] - ref['cost_mean']) / ref['cost_mean']
                half = 100 * np.hypot((row['cost_ci_hi'] - row['cost_ci_lo']) / 2, ref_half) / ref['cost_mean']
                color = POPULATION_COLORS.get(population, '0.4')
                cost_ax.bar(x, rel, width=width * 0.9, color=color, alpha=0.9)
                cost_ax.errorbar(x, rel, yerr=half, color='black', capsize=1.8, lw=0.8)
        cost_ax.axhline(0.0, color='black', lw=0.8)
        cost_ax.set_xticks(range(len(enforcing)), [MECHANISM_LABELS.get(m, m) for m in enforcing], fontsize=7.5)
        cost_ax.set_title('Cost vs coordinated [%]', fontsize=9.5, pad=3)
        cost_ax.grid(axis='y', color='0.9', lw=0.6, zorder=-1)
        cost_ax.tick_params(labelsize=8)
        handles = [Line2D([], [], color=POPULATION_COLORS.get(p, '0.4'), lw=1.6, label=POPULATION_LABELS.get(p, p)) for p in populations]
        handles.append(Line2D([], [], color='black', ls=':', lw=0.9, label='budget'))
        fig.legend(handles=handles, loc='outside upper center', ncol=len(handles), fontsize=8.5, handlelength=2.2)
        if out_path is not None:
            save_figure(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_figure(fig: Figure, out_path: str | Path) -> Path:
    '''
    Save `fig` as a vector PDF with embedded TrueType fonts. Creation and modification dates are omitted, so the same
    inputs give byte-identical files.
    '''

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with rc_context(PAPER_RC):
            fig.savefig(
                temporary,
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.03,
                metadata={'Creator': CREATOR, 'CreationDate': None, 'ModDate': None},
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------------------
# Options and inputs of the figures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureSettings:
    '''
    Which figures to draw and which recorded entries they show. Defaults match the manuscript.
    '''

    figures: tuple[str, ...] = field(default_factory=lambda: FIGURE_NAMES)
    learned_field: str = 'y_avg'  # 'y_avg' (tail average) or 'y_last' (last iterate, as in the original notebooks)
    learned_seed: int = 0  # seed shown in the policy grids
    non_agg_labels: tuple[str, ...] = ('A', 'B', 'C', 'D')
    agg_labels: tuple[str, ...] = ('E', 'F', 'I', 'H')
    k11_panels: tuple[tuple[str, str], ...] = (
        ('B', '(B)'),
        ('D', '(D)'),
        ('K11', '{ {0}, {1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10}, {11} }'),
        ('P2', '{ {0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11} }'),
    )
    heatmap_labels: tuple[str, ...] = tuple('ABCDEFGHIJK')
    heatmap_range: tuple[float, float] = (0.0, 3.0)
    distance_key: str = 'seed_mean'  # entry of pairwise_{case}.npz: 'seed_mean' (mean over seeds) or 'seed0'
    finite_n_methods: tuple[str, ...] = ('A', 'B', 'D', 'E', 'F', 'G', 'K')  # each also draws its -LP counterpart if present
    finite_n_policies: tuple[str, ...] = ('projection', 'lp_update')
    dual_methods: tuple[str, ...] = ('B', 'C', 'D', 'E', 'F', 'G')
    bound_partitions: tuple[str, ...] = ('E', 'F', 'G', 'H', 'I', 'J', 'K')
    bound_value: str = 'B_best'  # column of bound_rows.csv used as the bound (best valid bound of the row)
    bound_regimes: tuple[tuple[str, str], ...] = (('lemma2', 'Lemma 1 regime'), ('grid2e-05', 'Experimental regime'))
    commons_scenario: str = 'base'
    commons_case: str = 'CJS'
    commons_N: int = 0  # 0: the largest N in the summary
    commons_tie_break: str = 'no_request'
    commons_populations: tuple[str, ...] = ('coord', 'indep', 'selfish')
    commons_mechanisms: tuple[str, ...] = ('M0', 'M1', 'M2')
    commons_stage_mechanism: str = 'M0'  # mechanism of the stage panels
    study_instance: str = 'K5T10'  # tables with K and T columns are filtered to this instance


@dataclass(frozen=True)
class FigureInputs:
    '''
    Input paths, relative to the results root. Each is tried as given and then as ``<stage>/summary/<file>``, where
    ``summarize`` writes by default. `lp_solutions_dir` is the directory of the per-job records of the stage.
    '''

    solutions_K5T10: str = 'learners/solutions_K5T10.npz'
    solutions_K11T30: str = 'learners/solutions_K11T30.npz'
    pairwise_CJS: str = 'learners/pairwise_CJS.npz'
    pairwise_SJS: str = 'learners/pairwise_SJS.npz'
    lp_solutions_dir: str = 'lp_solutions'
    finite_n_summary: str = 'finite_n/finite_n_summary.csv'
    bound_rows: str = 'bound/bound_rows.csv'
    commons_summary: str = 'commons/commons_summary.csv'
    commons_stages: str = 'commons/commons_stages.csv'


_SETTINGS_FIELDS = {f.name for f in fields(FigureSettings)}
_INPUT_FIELDS = {f.name for f in fields(FigureInputs)}


def _tuple_of_str(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f'{name} must be a list of strings')
    return tuple(str(item) for item in value)


def options_from_mapping(mapping: Mapping[str, Any]) -> tuple[FigureSettings, FigureInputs]:
    '''
    Settings and inputs from a config mapping (the TOML of configs/figures.toml, or the stage's config).
    `results_root` and `out_dir` are not options here; unknown keys are refused, so typos do not pass silently.
    '''

    body = {key: value for key, value in mapping.items() if key not in ('results_root', 'out_dir', 'smoke')}
    inputs_table = dict(body.pop('inputs', {}) or {})
    unknown = sorted(set(body) - _SETTINGS_FIELDS)
    if unknown:
        raise ValueError(f'unknown figure option(s): {", ".join(unknown)}')
    unknown_inputs = sorted(set(inputs_table) - _INPUT_FIELDS)
    if unknown_inputs:
        raise ValueError(f'unknown figure input(s): {", ".join(unknown_inputs)}')

    values: dict[str, Any] = {}
    for key, value in body.items():
        if key == 'figures':
            values[key] = _tuple_of_str(value, key)
        elif key in ('learned_seed', 'commons_N'):
            values[key] = int(value)
        elif key == 'heatmap_range':
            low, high = value
            values[key] = (float(low), float(high))
        elif key == 'k11_panels':
            panels = []
            for item in value:
                if isinstance(item, Mapping):
                    panels.append((str(item['method']), str(item['title'])))
                else:
                    method, title = item
                    panels.append((str(method), str(title)))
            values[key] = tuple(panels)
        elif key == 'bound_regimes':
            pairs = []
            for item in value:
                if isinstance(item, Mapping):
                    pairs.append((str(item['regime']), str(item['title'])))
                else:
                    regime, title = item
                    pairs.append((str(regime), str(title)))
            values[key] = tuple(pairs)
        elif key in ('learned_field', 'distance_key', 'bound_value', 'commons_scenario', 'commons_case', 'commons_tie_break',
                     'commons_stage_mechanism', 'study_instance'):
            values[key] = str(value)
        else:
            values[key] = _tuple_of_str(value, key)
    settings = FigureSettings(**values)
    if settings.learned_field not in ('y_avg', 'y_last'):
        raise ValueError("learned_field must be 'y_avg' or 'y_last'")
    if len(settings.k11_panels) != 4:
        raise ValueError('k11_panels needs exactly four panels')
    if settings.study_instance not in INSTANCE_SHAPES:
        raise ValueError(f'study_instance must be one of {sorted(INSTANCE_SHAPES)}')
    if settings.distance_key not in ('seed_mean', 'seed0'):
        raise ValueError("distance_key must be 'seed_mean' or 'seed0'")
    inputs = FigureInputs(**{key: str(value) for key, value in inputs_table.items()})
    return settings, inputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_result_path(results_root: str | Path, relative: str) -> list[Path]:
    '''
    Candidate paths of an input: ``<root>/<relative>``, then ``<root>/<stage>/summary/<name>`` for a two-part path.
    '''

    rel = Path(relative)
    candidates = [Path(results_root) / rel]
    if len(rel.parts) == 2:
        candidates.append(Path(results_root) / rel.parts[0] / 'summary' / rel.parts[1])
    return candidates


def _study_KT(instance: str) -> tuple[int, int]:
    match = re.fullmatch(r'K(\d+)T(\d+)', instance)
    if match is None:
        raise ValueError(f'study_instance must look like K5T10, got {instance!r}')
    return int(match.group(1)), int(match.group(2))


# ---------------------------------------------------------------------------
# Loading recorded results
# ---------------------------------------------------------------------------


class FigureContext:
    '''
    Reads the recorded inputs of the figures from a results root, once each, and records which files a figure used.
    '''

    def __init__(self, results_root: str | Path, settings: FigureSettings, inputs: FigureInputs) -> None:
        self.results_root = Path(results_root)
        self.settings = settings
        self.inputs = inputs
        self.current = ''  # name of the figure being drawn; the files it reads are recorded under it
        self.used: dict[str, dict[str, Path]] = {}
        self._cache: dict[str, Any] = {}
        self._paths: dict[str, Path | None] = {}
        self._lp_records: dict[tuple[str, str, str, str], list[Path]] | None = None

    def _find(self, key: str) -> Path | None:
        for path in resolve_result_path(self.results_root, getattr(self.inputs, key)):
            if path.is_file() or path.is_dir():
                return path
        return None

    def _describe(self, key: str) -> str:
        return ' or '.join(str(path) for path in resolve_result_path(self.results_root, getattr(self.inputs, key)))

    def _record(self, label: str, path: Path) -> None:
        self.used.setdefault(self.current, {})[label] = path

    def npz(self, key: str) -> dict[str, np.ndarray] | None:
        '''
        All arrays of the input `key` (an .npz file), or None when the file is absent.
        '''

        if key not in self._cache:
            path = self._find(key)
            self._paths[key] = path
            if path is None:
                self._cache[key] = None
            else:
                with np.load(path, allow_pickle=False) as data:
                    self._cache[key] = {name: np.asarray(data[name]) for name in data.files}
        if self._paths.get(key) is not None:
            self._record(key, self._paths[key])
        return self._cache[key]

    def csv(self, key: str, schema: str) -> pd.DataFrame:
        '''
        The CSV input `key`, checked against the stage's schema. Raises if the file or a column is missing.
        '''

        if key not in self._cache:
            path = self._find(key)
            self._paths[key] = path
            if path is None:
                raise FigureInputError(f'missing input {key!r}: none of {self._describe(key)} exists')
            frame = pd.read_csv(path)
            missing = [column for column in FILE_SCHEMAS[schema] if column not in frame.columns]
            if missing:
                raise FigureInputError(f'{path}: missing columns {missing}; found {list(frame.columns)}')
            self._cache[key] = frame
        self._record(key, self._paths[key])
        return self._cache[key]

    def _lp_index(self) -> dict[tuple[str, str, str, str], list[Path]]:
        '''
        The per-job records of the lp_solutions stage that finished, keyed by (instance, case, kind, partition).
        '''

        if self._lp_records is None:
            from lplearn.experiments.stages.lp_solutions import VERSION as LP_VERSION  # records of the current code only

            directory = self._find('lp_solutions_dir')
            index: dict[tuple[str, str, str, str], list[Path]] = {}
            if directory is not None and directory.is_dir():
                for path in sorted(directory.glob('*.json')):
                    try:
                        record = json.loads(path.read_text(encoding='utf-8'))
                    except (OSError, ValueError):
                        continue
                    job = record.get('job') if isinstance(record, dict) else None
                    if not isinstance(job, dict) or record.get('status') != 'ok':
                        continue
                    key = (str(job.get('instance')), str(job.get('case')), str(job.get('kind')), str(job.get('partition')))
                    current = str(record.get('stage_version')) == str(LP_VERSION)
                    index.setdefault(key, []).append((current, path))
            # Records of superseded stage versions are kept on disk; use the current ones when a key has any.
            self._lp_records = {
                key: [path for current, path in items if current] or [path for _, path in items]
                for key, items in index.items()
            }
        return self._lp_records

    def exact(self, instance: str, case: str, label: str) -> dict[str, np.ndarray]:
        '''
        The interior-point solution of exact label A (kind lp) or B (kind lp_reg) on the detailed grid:
        arrays ``y`` (T, S, 4) and ``lam_res`` (2, T).
        '''

        kind, partition = _exact_kind_partition(label)
        paths = self._lp_index().get((instance, case, kind, partition), [])
        if len(paths) != 1:
            where = self._describe('lp_solutions_dir')
            raise FigureInputError(
                f'expected one lp_solutions record for {case}/{label} on {instance} (kind {kind!r}, partition {partition!r}), '
                f'found {len(paths)} in {where}'
            )
        npz_path = paths[0].with_suffix('.npz')
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name], dtype=float) for name in ('y', 'lam_res')}
        self._record(f'lp_solutions/{instance}/{case}/{kind}', paths[0])
        self._record(f'lp_solutions/{instance}/{case}/{kind}.npz', npz_path)
        return arrays

    def policy(self, instance: str, case: str, label: str, title: str) -> PolicyPanel:
        '''
        The occupation measure of method `label` for `case` on `instance`: A and B come from the lp_solutions records,
        learned labels from the learners solutions at `learned_field` and `learned_seed`.
        '''

        if label in EXACT_LABELS or label.endswith('-LP'):
            arrays = self.exact(instance, case, label)
            y = arrays['y']
            _check_instance_shape(y, instance, f'{case}/{label}')
            kind, partition = _exact_kind_partition(label)
            return PolicyPanel(title=title, y=y, source=f'lp_solutions {kind} {partition} {case}')
        field = self.settings.learned_field
        seed = self.settings.learned_seed
        name = f'{case}/{label}/seed{seed}/{field}'
        store = self.npz(f'solutions_{instance}')
        if store is None or name not in store:
            where = self._describe(f'solutions_{instance}')
            raise FigureInputError(f'no {name!r} in {where}')
        y = np.asarray(store[name], dtype=float)
        _check_instance_shape(y, instance, name)
        return PolicyPanel(title=title, y=y, source=name)

    def dual_prices(self, instance: str, case: str, label: str) -> tuple[np.ndarray, np.ndarray]:
        '''
        Mean and standard deviation over seeds of lambda_res[l, t] (shape (2, T)). Learned labels use the lam_res field
        that matches `learned_field`; exact labels use the single lp_solutions run (standard deviation zero).
        '''

        T = INSTANCE_SHAPES[instance][0]
        if label in EXACT_LABELS:
            lam = self.exact(instance, case, label)['lam_res']
            if lam.shape != (2, T):
                raise FigureInputError(f'lam_res of {case}/{label} has shape {lam.shape}, expected (2, {T})')
            return lam, np.zeros_like(lam)
        suffix = self.settings.learned_field.split('_', 1)[1]
        store = self.npz(f'solutions_{instance}')
        if store is None:
            raise FigureInputError(f'missing input solutions_{instance}: none of {self._describe(f"solutions_{instance}")} exists')
        pattern = re.compile(rf'{re.escape(case)}/{re.escape(label)}/seed(\d+)/lam_res_{suffix}')
        runs = {int(m.group(1)): np.asarray(arr, dtype=float) for name, arr in store.items() if (m := pattern.fullmatch(name))}
        if not runs:
            raise FigureInputError(f'no lam_res_{suffix} for {case}/{label} on {instance}')
        stack = np.stack([runs[seed] for seed in sorted(runs)])
        if stack.shape[1:] != (2, T):
            raise FigureInputError(f'lam_res of {case}/{label} has shape {stack.shape[1:]}, expected (2, {T})')
        sd = stack.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(stack.shape[1:])
        return stack.mean(axis=0), sd

    def distance_matrix(self, case: str) -> np.ndarray:
        '''
        Pairwise Frobenius distances of `case` (learners' pairwise_{case}.npz, entry `distance_key`), ordered by
        `heatmap_labels`.
        '''

        store = self.npz(f'pairwise_{case}')
        if store is None:
            raise FigureInputError(f'missing input pairwise_{case}: none of {self._describe(f"pairwise_{case}")} exists')
        for name in ('labels', self.settings.distance_key):
            if name not in store:
                raise FigureInputError(f'pairwise_{case}: missing entry {name!r}; found {sorted(store)}')
        labels = [str(label) for label in store['labels']]
        matrix = np.asarray(store[self.settings.distance_key], dtype=float)
        if matrix.shape != (len(labels), len(labels)):
            raise FigureInputError(f'pairwise_{case}: {self.settings.distance_key} has shape {matrix.shape} for {len(labels)} labels')
        index = {label: i for i, label in enumerate(labels)}
        wanted = list(self.settings.heatmap_labels)
        missing = [label for label in wanted if label not in index]
        if missing:
            raise FigureInputError(f'pairwise_{case}: labels {missing} are not in the matrix (it has {labels})')
        sel = [index[label] for label in wanted]
        return matrix[np.ix_(sel, sel)]

    def finite_n_frame(self) -> pd.DataFrame:
        frame = self.csv('finite_n_summary', 'finite_n_summary')
        return frame[frame['case'].astype(str).isin(CASES)]

    def bound_frame(self) -> pd.DataFrame:
        '''
        Rows of the study instance with the columns of ``PLOT_SCHEMAS['bound']``. The true error is ``dist_total``
        (||w* - w_bar*||), the bound is ``bound_value`` and the partition is the row's ``label``.
        '''

        raw = self.csv('bound_rows', 'bound_rows')
        K, T = _study_KT(self.settings.study_instance)
        raw = raw[(raw['K'].astype(int) == K) & (raw['T'].astype(int) == T)]
        regimes = [regime for regime, _ in self.settings.bound_regimes]
        raw = raw[raw['regime'].astype(str).isin(regimes)]
        if self.settings.bound_value not in raw.columns:
            raise FigureInputError(f'bound_rows: missing column {self.settings.bound_value!r}')
        frame = pd.DataFrame({
            'case': raw['case'].astype(str).to_numpy(),
            'partition': raw['label'].astype(str).to_numpy(),
            'convention': raw['convention'].astype(str).to_numpy(),
            'regime': raw['regime'].astype(str).to_numpy(),
            'true_error': raw['dist_total'].astype(float).to_numpy(),
            'bound': raw[self.settings.bound_value].astype(float).to_numpy(),
        })
        return frame[frame['case'].isin(CASES)]

    def commons_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        The cost table (one row per mechanism and population) and the stage table of the no-coordinator experiment.
        '''

        summary = self.csv('commons_summary', 'commons_summary')
        s = self.settings
        sub = summary[(summary['scenario'].astype(str) == s.commons_scenario) & (summary['case'].astype(str) == s.commons_case)
                      & (summary['tie_break'].astype(str) == s.commons_tie_break)]
        if sub.empty:
            raise FigureInputError(f'commons_summary: no rows for scenario {s.commons_scenario!r}, case {s.commons_case!r}, '
                                   f'tie_break {s.commons_tie_break!r}')
        N = int(sub['N'].max()) if s.commons_N == 0 else s.commons_N
        sub = sub[sub['N'].astype(int) == N]
        sub = sub[sub['population'].astype(str).isin(s.commons_populations) & sub['mechanism'].astype(str).isin(s.commons_mechanisms)]
        costs = pd.DataFrame({
            'mechanism': sub['mechanism'].astype(str).to_numpy(),
            'population': sub['population'].astype(str).to_numpy(),
            'cost_mean': sub['cost_mean'].astype(float).to_numpy(),
            'cost_ci_lo': (sub['cost_mean'] - sub['cost_ci95']).astype(float).to_numpy(),
            'cost_ci_hi': (sub['cost_mean'] + sub['cost_ci95']).astype(float).to_numpy(),
        })
        stages = self.csv('commons_stages', 'commons_stages')
        stage_sub = stages[(stages['case'].astype(str) == s.commons_case) & (stages['N'].astype(int) == N)
                           & (stages['tie_break'].astype(str) == s.commons_tie_break)
                           & (stages['mechanism'].astype(str) == s.commons_stage_mechanism)
                           & stages['population'].astype(str).isin(s.commons_populations)]
        return costs, stage_sub[list(PLOT_SCHEMAS['commons_stages'])].copy()


def _check_instance_shape(y: np.ndarray, instance: str, name: str) -> None:
    T, S = INSTANCE_SHAPES[instance]
    if y.shape != (T, S, 4):
        raise FigureInputError(
            f'{name} has shape {y.shape}; expected (T, S, 4) = ({T}, {S}, 4) in the core layout y[t, s, u], u = 2a + b'
        )
    if not np.all(np.isfinite(y)):
        raise FigureInputError(f'{name} contains non-finite values')


# ---------------------------------------------------------------------------
# Figure builders: each reads its inputs from the context and returns a Figure
# ---------------------------------------------------------------------------


def _fig_policy_solutions_non_agg(ctx: FigureContext) -> Figure:
    s = ctx.settings
    cjs = [ctx.policy('K5T10', 'CJS', label, f'CJS Solution ({label})') for label in s.non_agg_labels]
    sjs = [ctx.policy('K5T10', 'SJS', label, f'SJS Solution ({label})') for label in s.non_agg_labels]
    return plot_policy_solutions_non_agg(cjs, sjs)


def _fig_policy_solutions_agg(ctx: FigureContext) -> Figure:
    s = ctx.settings
    cjs = [ctx.policy('K5T10', 'CJS', label, f'CJS Solution ({label})') for label in s.agg_labels]
    sjs = [ctx.policy('K5T10', 'SJS', label, f'SJS Solution ({label})') for label in s.agg_labels]
    return plot_policy_solutions_agg(cjs, sjs)


def _fig_heatmap(ctx: FigureContext) -> Figure:
    labels = ctx.settings.heatmap_labels
    return plot_heatmap_policies_compare(labels, ctx.distance_matrix('CJS'), ctx.distance_matrix('SJS'),
                                         value_range=ctx.settings.heatmap_range)


def _fig_policies_K11T30(ctx: FigureContext) -> Figure:
    panels = [ctx.policy('K11T30', 'CJS', label, title) for label, title in ctx.settings.k11_panels]
    return plot_policies_K11T30(panels)


def _fig_finite_n_costs(ctx: FigureContext) -> Figure:
    s = ctx.settings
    return plot_finite_n_costs(ctx.finite_n_frame(), methods=s.finite_n_methods, policies=s.finite_n_policies)


def _fig_bound_validation(ctx: FigureContext) -> Figure:
    s = ctx.settings
    return plot_bound_validation(ctx.bound_frame(), partitions=s.bound_partitions, regimes=s.bound_regimes)


def _fig_dual_prices(ctx: FigureContext) -> Figure:
    rows = []
    for case in CASES:
        for label in ctx.settings.dual_methods:
            mean, sd = ctx.dual_prices('K5T10', case, label)
            for l_index, (resource, _) in enumerate(RESOURCES):
                for t in range(mean.shape[1]):
                    rows.append({'case': case, 'method': label, 'resource': resource, 't': t,
                                 'mean': float(mean[l_index, t]), 'sd': float(sd[l_index, t])})
    return plot_dual_prices(pd.DataFrame(rows), methods=ctx.settings.dual_methods)


def _fig_commons_overload(ctx: FigureContext) -> Figure:
    costs, stages = ctx.commons_frames()
    s = ctx.settings
    return plot_commons_overload(stages, costs, populations=s.commons_populations, mechanisms=s.commons_mechanisms)


FIGURE_BUILDERS: dict[str, Callable[[FigureContext], Figure]] = {
    'policy_solutions_non_agg': _fig_policy_solutions_non_agg,
    'policy_solutions_agg': _fig_policy_solutions_agg,
    'heatmap_policies_compare': _fig_heatmap,
    'policies_K11T30': _fig_policies_K11T30,
    'finite_n_costs': _fig_finite_n_costs,
    'bound_validation': _fig_bound_validation,
    'dual_prices': _fig_dual_prices,
    'commons_overload': _fig_commons_overload,
}
FIGURE_NAMES: tuple[str, ...] = tuple(FIGURE_BUILDERS)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class FigureRun:
    '''
    What a call to ``build_figures`` produced: written files and inputs per figure, and the failures with their reasons.
    '''

    written: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    tracebacks: dict[str, str] = field(default_factory=dict)  # only for unexpected errors (not missing inputs)
    preserved: dict[str, dict[str, Any]] = field(default_factory=dict)  # prior PDF, not generated by this run


def build_figures(results_root: str | Path, out_dir: str | Path, *, names: Sequence[str] | None = None,
                  settings: FigureSettings | None = None, inputs: FigureInputs | None = None) -> FigureRun:
    '''
    Draw the requested figures from the recorded results and write ``<out_dir>/<name>.pdf``.

    A figure whose inputs are missing or malformed is reported in ``failed``; existing PDFs are preserved and marked
    as prior output, not as an output of this run. The other figures are still drawn.
    '''

    settings = settings or FigureSettings()
    inputs = inputs or FigureInputs()
    requested = tuple(names) if names is not None else (settings.figures or FIGURE_NAMES)
    unknown = [name for name in requested if name not in FIGURE_BUILDERS]
    if unknown:
        raise ValueError(f'unknown figure(s) {unknown}; known: {", ".join(FIGURE_NAMES)}')
    out = Path(out_dir)
    ctx = FigureContext(results_root, settings, inputs)
    run = FigureRun()
    for name in requested:
        ctx.current = name
        try:
            figure = FIGURE_BUILDERS[name](ctx)
            path = save_figure(figure, out / f'{name}.pdf')
        except Exception as exc:  # reported in the run; the other figures continue
            run.failed[name] = f'{type(exc).__name__}: {exc}'
            if not isinstance(exc, FigureInputError):
                run.tracebacks[name] = traceback.format_exc()
            previous = out / f'{name}.pdf'
            if previous.is_file():
                run.preserved[name] = {'path': str(previous), 'sha256': _sha256(previous),
                                       'bytes': previous.stat().st_size}
            continue
        finally:
            figure = None
        run.written[name] = {
            'path': str(path),
            'sha256': _sha256(path),
            'bytes': path.stat().st_size,
            'inputs': [{'label': label, 'path': str(file), 'sha256': _sha256(file)}
                       for label, file in sorted(ctx.used.get(name, {}).items())],
        }
    return run


def manifest(run: FigureRun, *, results_root: str | Path, out_dir: str | Path, settings: FigureSettings,
             inputs: FigureInputs) -> dict[str, Any]:
    '''
    JSON-serialisable provenance of a run: settings, inputs, and per figure the output hash and the input files used.
    '''

    return {
        'schema': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'results_root': str(results_root),
        'out_dir': str(out_dir),
        'settings': asdict(settings),
        'inputs': asdict(inputs),
        'written': run.written,
        'failed': run.failed,
        'preserved': run.preserved,
        'tracebacks': run.tracebacks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    '''
    Command line: draw the figures from a results root. Exit codes: 0 all written, 1 some figures failed, 2 bad arguments.
    '''

    import argparse
    import sys
    import tomllib

    parser = argparse.ArgumentParser(prog='python -m lplearn.visualizations.paper_figures',
                                     description='Draw the SIGMETRICS figures from recorded results.')
    parser.add_argument('--results-root', default=None, help=f'results root (default: {DEFAULT_RESULTS_ROOT}, or the config value)')
    parser.add_argument('--out-dir', default=None, help=f'output directory (default: {DEFAULT_OUT_DIR}, or the config value)')
    parser.add_argument('--only', action='append', choices=FIGURE_NAMES, metavar='NAME',
                        help='figure to draw (repeatable; default: the figures of the config, or all)')
    parser.add_argument('--config', type=Path, default=None, help='TOML with the figure options and [inputs] (default: built-in)')
    parser.add_argument('--manifest', type=Path, default=None, help='also write the provenance manifest (JSON) to this path')
    args = parser.parse_args(argv)

    mapping: dict[str, Any] = {}
    if args.config is not None:
        try:
            mapping = tomllib.loads(args.config.read_text(encoding='utf-8'))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(f'paper_figures: cannot read {args.config}: {exc}', file=sys.stderr)
            return 2
    try:
        settings, inputs = options_from_mapping(mapping)
    except (ValueError, KeyError, TypeError) as exc:
        print(f'paper_figures: invalid options: {exc}', file=sys.stderr)
        return 2
    results_root = args.results_root or mapping.get('results_root', DEFAULT_RESULTS_ROOT)
    out_dir = args.out_dir or mapping.get('out_dir', DEFAULT_OUT_DIR)
    names = args.only or (settings.figures or FIGURE_NAMES)
    run = build_figures(results_root, out_dir, names=names, settings=settings, inputs=inputs)
    for name, info in run.written.items():
        print(f'wrote {info["path"]}')
    for name, message in run.failed.items():
        print(f'not written: {name}: {message}', file=sys.stderr)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest(run, results_root=results_root, out_dir=out_dir, settings=settings,
                                                     inputs=inputs), indent=2, sort_keys=True, default=str) + '\n',
                                 encoding='utf-8')
    return 1 if run.failed else 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
