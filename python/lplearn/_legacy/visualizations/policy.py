'''
This module contains functions to visualize the policy solution.
'''


import numpy as np
import matplotlib.pyplot as plt
import warnings
from lplearn._legacy.utils import palettes, area_from_radius, radius_from_area


def policy_pie_plot(
        y: np.ndarray,
        min_radius: float = .1,
        max_radius: float = .7,
        fontsize: int = None,
        label_fontsize: int = None,
        fontfamily: str = 'Times New Roman',
        color_palette: dict = dict(palettes.DEFAULT),
        min_value: float = 1e-2,
        show_legend: bool = False,
        ax: plt.Axes = None
    ) -> tuple[plt.Figure, plt.Axes]:

    '''
    Plot the policy as a pie chart.
    The policy is a 4D tensor with dimensions (K+1, 2, 2, T), where:
    - K is the buffer size,
    - T is the number of time slots,
    - 2 is the number of actions a,
    - 2 is the number of actions b.
    Non-finite values are rejected. A warning is emitted for materially
    negative values or stage totals that differ from one. These checks do not
    verify the model's flow or resource constraints. The plot may modify input.
    '''

    # Warn before the legacy clipping/normalization can conceal invalid input.
    # A mass check is a diagnostic, not a proof of feasibility.
    finite = np.isfinite(y)
    invalid = []
    if not np.all(finite):
        raise ValueError("Policy plot input contains non-finite occupations")
    if np.any(y[finite] < -1e-5):
        invalid.append("negative occupations")
    stage_mass = np.sum(y, axis=(0, 1, 2))
    if not np.all(np.isfinite(stage_mass) & np.isclose(stage_mass, 1.0, atol=1e-5, rtol=0)):
        invalid.append("stage mass differs from one")
    if invalid:
        warnings.warn(
            "Policy plot input has " + ", ".join(invalid)
            + "; legacy clipping and normalization may hide these defects. "
            "Model flow and resource constraints were not checked.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Get the dimensions of the policy tensor
    T = y.shape[-1]
    K = y.shape[0] - 1

    # Set negative values to zero and round the values
    y[y < min_value] = 0  # Set negative values to zero

    # Create the figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(T*2, K*2), dpi=50)
    else:
        fig = ax.figure  # Get the figure from the provided axis
    fontsize = fontsize if fontsize else np.sqrt(K*T)*4
    label_fontsize = label_fontsize if label_fontsize else fontsize*1.5

    for t in range(T-1, -1, -1):

        masses = np.array([y[s,:,:,t].sum() for s in range(K+1)])
        masses[masses < min_value] = 0
        total_mass = masses.sum()
        normalized_masses = np.array([np.round(mass/total_mass, 2) for mass in masses])
        positive_masses = normalized_masses[normalized_masses > 0]
        if positive_masses.size > 0:
            min_norm_masses = np.min(positive_masses)
        else:
            continue
        down_scaled_areas = np.array([area_from_radius(max_radius) * mass for mass in normalized_masses])
        up_scaled_areas = np.array([area_from_radius(min_radius) * (mass/min_norm_masses) for mass in normalized_masses])
        min_down_areas = np.min(down_scaled_areas[down_scaled_areas > 0])

        # if min_down_areas < area_from_radius(min_radius) and max(up_scaled_areas) <= area_from_radius(max_radius):
        #     scaled_areas = np.array([(down_scale + up_scale) / 2 for down_scale, up_scale in zip(down_scaled_areas, up_scaled_areas)])
        # else:
        #     scaled_areas = down_scaled_areas
        scaled_areas = down_scaled_areas

        scaled_radius = np.array([radius_from_area(area) for area in scaled_areas])
        for s in range(K, -1, -1):
            if normalized_masses[s] > 0:
                ax.pie(
                    x = y[s,:,:,t].flatten(),
                    radius = scaled_radius[s],
                    center = (t, s),
                    startangle = 90,
                    colors = [color_palette['a0b0'], color_palette['a0b1'], color_palette['a1b0'], color_palette['a1b1']])

    ax.set_xlim(-1.1*max_radius, y.shape[-1])
    ax.set_ylim(-1.1*max_radius, y.shape[0]+.5)
    ax.set_xticks(range(y.shape[-1]))
    ax.set_yticks(range(y.shape[0]))
    ax.tick_params(axis='x', labelsize=fontsize, labelfontfamily=fontfamily)
    ax.tick_params(axis='y', labelsize=fontsize, labelfontfamily=fontfamily)
    ax.set_xlabel('Arrival time (t)', fontsize=label_fontsize, fontfamily=fontfamily)  # set_xlabel | set_xticklabels
    ax.set_ylabel('Queue size (s)', fontsize=label_fontsize, fontfamily=fontfamily)

    # Legend
    if show_legend:
        labels = [r'$\tilde{y}_{s,0,0}(t)$', r'$\tilde{y}_{s,0,1}(t)$', r'$\tilde{y}_{s,1,0}(t)$', r'$\tilde{y}_{s,1,1}(t)$']
        legend = ax.legend(labels, loc='upper center', frameon=True, ncols=4, bbox_to_anchor=(.5, .98), prop={'size': fontsize*2, 'family': 'Times New Roman'})
        frame = legend.get_frame()
        frame.set_edgecolor('black')  # Set legend border color
        frame.set_linewidth(.5)
    # plt.tight_layout()

    return fig, ax
