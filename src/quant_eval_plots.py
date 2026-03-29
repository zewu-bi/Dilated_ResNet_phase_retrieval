
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from src.prab_plot_style import PRAB_COLORS, setup_prab_style
from src.quant_eval_utils import METHOD_GS, METHOD_NN, GROUP_ORDER, finite_array


COLORS = {
    "target": PRAB_COLORS["target"],
    METHOD_NN: PRAB_COLORS["nn"],
    METHOD_GS: PRAB_COLORS["gs"],
}


def plot_profile_pair(
    ax,
    z_um,
    target,
    prediction,
    target_label="Target",
    prediction_label="Prediction",
    target_color=None,
    prediction_color=None,
    prediction_linestyle="--",
    xlabel=None,
    ylabel=None,
    title=None,
    legend_loc="upper right",
    grid=True,
):
    setup_prab_style()
    target_color = target_color or COLORS["target"]
    prediction_color = prediction_color or COLORS[METHOD_NN]

    ax.plot(z_um, target, linewidth=2, color=target_color, label=target_label)
    ax.plot(z_um, prediction, linewidth=2, linestyle=prediction_linestyle, color=prediction_color, label=prediction_label)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    if legend_loc is not None:
        ax.legend(loc=legend_loc)
    return ax


def plot_spectrum_pair(
    ax,
    freq_meas,
    measured_band,
    freq_full,
    reconstructed_full,
    f_min,
    f_max,
    f_display_max,
    measured_label="Measured spectrum",
    reconstructed_label="Forward spectrum",
    measured_color=None,
    reconstructed_color=None,
    reconstructed_linestyle="--",
    shade_color="blue",
    shade_alpha=0.08,
    xlabel=None,
    ylabel=None,
    title=None,
    legend_loc="upper right",
    normalize_mode="band-max",
    grid=True,
):
    setup_prab_style()
    measured_color = measured_color or COLORS["target"]
    reconstructed_color = reconstructed_color or COLORS[METHOD_NN]

    measured_band = np.asarray(measured_band, dtype=float)
    reconstructed_full = np.asarray(reconstructed_full, dtype=float)
    freq_meas = np.asarray(freq_meas, dtype=float)
    freq_full = np.asarray(freq_full, dtype=float)

    if normalize_mode == "band-max":
        mask_band = (freq_full >= f_min) & (freq_full <= f_max)
        scale_forward = np.max(reconstructed_full[mask_band]) if np.any(mask_band) else np.max(reconstructed_full)
        scale_measured = np.max(measured_band)
    elif normalize_mode == "global-max":
        scale_forward = np.max(reconstructed_full)
        scale_measured = np.max(measured_band)
    else:
        scale_forward = 1.0
        scale_measured = 1.0

    scale_forward = scale_forward if scale_forward > 0 else 1.0
    scale_measured = scale_measured if scale_measured > 0 else 1.0

    ax.axvspan(0, f_min, color=shade_color, alpha=shade_alpha)
    ax.axvspan(f_max, f_display_max, color=shade_color, alpha=shade_alpha)
    ax.plot(freq_meas, measured_band / scale_measured, linewidth=2, color=measured_color, label=measured_label)
    ax.plot(
        freq_full,
        reconstructed_full / scale_forward,
        linewidth=2,
        linestyle=reconstructed_linestyle,
        color=reconstructed_color,
        label=reconstructed_label,
    )
    ax.set_xlim(0, f_display_max)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    if legend_loc is not None:
        ax.legend(loc=legend_loc)
    return ax


def plot_profile_envelope(
    ax,
    z_um,
    target,
    nn_prediction,
    gs_quantiles=None,
    gs_stack=None,
    target_label="Target",
    nn_label=METHOD_NN,
    gs_label=None,
    target_color=None,
    nn_color=None,
    gs_color=None,
    nn_linestyle="--",
    envelope_alpha=0.18,
    xlabel=None,
    ylabel=None,
    title=None,
    legend_loc="upper right",
    grid=True,
):
    setup_prab_style()
    target_color = target_color or COLORS["target"]
    nn_color = nn_color or COLORS[METHOD_NN]
    gs_color = gs_color or COLORS[METHOD_GS]

    ax.plot(z_um, target, color=target_color, label=target_label)
    ax.plot(z_um, nn_prediction, linestyle=nn_linestyle, color=nn_color, label=nn_label)

    if gs_quantiles is not None:
        if gs_label is None:
            gs_label = "GS envelope"
        ax.fill_between(z_um, gs_quantiles["q05"], gs_quantiles["q95"], color=gs_color, alpha=envelope_alpha, label=gs_label)
    elif gs_stack is not None and len(gs_stack) > 0:
        if gs_label is None:
            gs_label = "GS envelope"
        gs_stack = np.asarray(gs_stack, dtype=float)
        ax.fill_between(z_um, np.min(gs_stack, axis=0), np.max(gs_stack, axis=0), color=gs_color, alpha=envelope_alpha, label=gs_label)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    if legend_loc is not None:
        ax.legend(loc=legend_loc)
    return ax


def grouped_method_boxplot(
    ax,
    data_by_key,
    groups,
    methods,
    colors,
    xticklabels=None,
    ylabel=None,
    title=None,
    show_legend=True,
    legend_loc="upper right",
):
    setup_prab_style()

    positions, data, facecolors = [], [], []
    base_positions = np.arange(len(groups)) * 3.0
    offsets = np.linspace(-0.45, 0.45, len(methods))

    for gi, group in enumerate(groups):
        for mi, method in enumerate(methods):
            vals = finite_array(data_by_key[(group, method)])
            positions.append(base_positions[gi] + offsets[mi])
            data.append(vals if len(vals) > 0 else np.array([np.nan]))
            facecolors.append(colors[method])

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.75,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )
    for patch, fc in zip(bp["boxes"], facecolors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.65)

    ax.set_xticks(base_positions)
    ax.set_xticklabels(list(xticklabels) if xticklabels is not None else list(groups))
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    if show_legend:
        handles = [Patch(facecolor=colors[m], edgecolor="black", alpha=0.65, label=m) for m in methods]
        ax.legend(handles=handles, loc=legend_loc)
    return ax


def paired_metric_boxplot(
    ax_left,
    ax_right,
    left_values,
    right_values,
    method_order,
    colors,
    xticklabels=("Metric A", "Metric B"),
    left_ylabel=None,
    right_ylabel=None,
    title=None,
    legend_loc="upper right",
):
    setup_prab_style()

    x_left = [0.85, 1.15]
    x_right = [1.85, 2.15]

    _two_method_boxplot(ax_left, left_values[0], left_values[1], positions=x_left, colors=[colors[m] for m in method_order])
    _two_method_boxplot(ax_right, right_values[0], right_values[1], positions=x_right, colors=[colors[m] for m in method_order])

    ax_left.set_xlim(0.45, 2.55)
    ax_left.set_xticks([1.0, 2.0])
    ax_left.set_xticklabels(list(xticklabels))
    if left_ylabel is not None:
        ax_left.set_ylabel(left_ylabel)
    if right_ylabel is not None:
        ax_right.set_ylabel(right_ylabel)
    if title is not None:
        ax_left.set_title(title)
    ax_left.grid(axis="y", alpha=0.25)

    handles = [Patch(facecolor=colors[m], edgecolor="black", alpha=0.65, label=m) for m in method_order]
    ax_left.legend(handles=handles, loc=legend_loc)
    return ax_left, ax_right


def _two_method_boxplot(ax, values_method_1, values_method_2, positions, colors, widths=0.22):
    safe_data = [
        finite_array(values_method_1) if len(finite_array(values_method_1)) else np.array([np.nan]),
        finite_array(values_method_2) if len(finite_array(values_method_2)) else np.array([np.nan]),
    ]
    bp = ax.boxplot(
        safe_data,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )
    for patch, fc in zip(bp["boxes"], colors):
        patch.set_facecolor(fc)
        patch.set_alpha(0.65)
    return bp


def plot_interpolated_heatmap(
    ax,
    heatmap_data,
    fig=None,
    title=None,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    cmap="bwr",
    show_points=False,
    point_color="k",
    point_size=10,
):
    setup_prab_style()

    x = np.asarray(heatmap_data["x"], dtype=float)
    y = np.asarray(heatmap_data["y"], dtype=float)
    z = np.asarray(heatmap_data["z"], dtype=float)

    if z.size == 0 or x.size == 0 or y.size == 0:
        ax.text(0.5, 0.5, "No finite double-bunch heatmap data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None, None

    finite = z[np.isfinite(z)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1e-12)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(
        z,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
    )

    if show_points:
        ax.scatter(
            heatmap_data.get("points_x", []),
            heatmap_data.get("points_y", []),
            s=point_size,
            c=point_color,
            alpha=0.35,
            linewidths=0.0,
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(False)

    cbar = None
    if fig is not None:
        cbar = fig.colorbar(im, ax=ax)
        if cbar_label is not None:
            cbar.set_label(cbar_label)

    return im, cbar


def plot_noise_profile_set(
    ax,
    diag,
    baseline_label="0 noise",
    rep_labels=None,
    baseline_color="black",
    rep_colors=None,
    rep_linestyles=None,
    xlabel=None,
    ylabel=None,
    title=None,
    legend_loc="upper right",
):
    setup_prab_style()

    rep_profiles = diag["rep_profiles"]
    if rep_colors is None:
        rep_colors = ["#1f77b4", "#e67e22", "#d62728"]
    if rep_linestyles is None:
        rep_linestyles = ["--", "-.", ":"]
    if rep_labels is None:
        rep_labels = [f"{nl * 100:.2f}% noise" for nl, _ in rep_profiles]

    ax.plot(diag["z_um"], diag["baseline_profile"], color=baseline_color, linestyle="-", linewidth=2.2, label=baseline_label)
    for i, ((_, prof), label) in enumerate(zip(rep_profiles, rep_labels)):
        ax.plot(
            diag["z_um"],
            prof,
            color=rep_colors[i % len(rep_colors)],
            linestyle=rep_linestyles[i % len(rep_linestyles)],
            linewidth=2.0,
            label=label,
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    if legend_loc is not None:
        ax.legend(loc=legend_loc)
    return ax


def plot_noise_error_boxes(
    ax,
    diag,
    xlabel="Noise level",
    ylabel="Peak-weighted relative error to zero-noise output",
    title=None,
    legend_loc="upper right",
    box_facecolor=None,
    median_color="black",
    baseline_color="black",
    xscale="log",
    xticks=None,
    xticklabels=None,
    curve_label="Median trend",
    zero_ref_label="Zero-noise reference",
):
    setup_prab_style()

    positions = np.asarray(diag["right_noise_levels"], dtype=float)
    all_errors = np.asarray(diag["all_errors_displayed"], dtype=float)
    medians = np.asarray(diag["median_displayed"], dtype=float)

    if box_facecolor is None:
        box_facecolor = COLORS[diag["method_label"]]

    if len(positions) == 1:
        widths = [0.25 * positions[0]] if xscale == "log" else [max(0.25 * positions[0], 1e-6)]
    elif xscale == "log":
        widths = 0.22 * positions
    else:
        diffs = np.diff(np.sort(positions))
        widths = [0.45 * float(np.min(diffs))] * len(positions)

    bp = ax.boxplot(
        [all_errors[i] for i in range(len(positions))],
        positions=positions,
        widths=widths,
        manage_ticks=False,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=median_color, linewidth=2),
        whiskerprops=dict(color=box_facecolor, linewidth=1.4),
        capprops=dict(color=box_facecolor, linewidth=1.4),
        boxprops=dict(color=box_facecolor, linewidth=1.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(box_facecolor)
        patch.set_alpha(0.45)

    ax.plot(positions, medians, color=median_color, linewidth=2.2, marker="o", markersize=4, label=curve_label)
    ax.axhline(0.0, color=baseline_color, linewidth=1.2, linestyle=":", label=zero_ref_label)

    ax.set_xscale(xscale)
    if xscale == "log":
        ax.set_xlim(positions.min() * 0.85, positions.max() * 1.18)
    else:
        if len(positions) == 1:
            pad = max(0.35 * positions[0], 1e-6)
        else:
            pad = 0.55 * np.min(np.diff(np.sort(positions)))
        ax.set_xlim(max(0.0, positions.min() - pad), positions.max() + pad)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    if legend_loc is not None:
        ax.legend(loc=legend_loc)
    return ax


def plot_noise_comparison(
    ax,
    diag_list,
    xlabel="Noise level",
    ylabel="Peak-weighted relative error to zero-noise output",
    title="Noise robustness comparison",
    legend_loc="upper left",
    xscale="log",
    xticks=None,
    xticklabels=None,
    curve_labels=None,
    curve_colors=None,
    curve_linestyles=None,
    show_boxes=True,
    box_alpha=0.16,
    show_zero_reference=True,
    baseline_color="black",
):
    setup_prab_style()

    if curve_labels is None:
        curve_labels = [diag["method_label"] for diag in diag_list]
    if curve_colors is None:
        curve_colors = [COLORS[diag["method_label"]] for diag in diag_list]
    if curve_linestyles is None:
        curve_linestyles = ["-", "--", "-.", ":"]

    for i, diag in enumerate(diag_list):
        positions = np.asarray(diag["right_noise_levels"], dtype=float)
        all_errors = np.asarray(diag["all_errors_displayed"], dtype=float)
        medians = np.asarray(diag["median_displayed"], dtype=float)
        color = curve_colors[i]
        linestyle = curve_linestyles[i % len(curve_linestyles)]

        if show_boxes:
            if len(positions) == 1:
                widths = [0.18 * positions[0]] if xscale == "log" else [max(0.18 * positions[0], 1e-6)]
            elif xscale == "log":
                widths = 0.14 * positions
            else:
                diffs = np.diff(np.sort(positions))
                widths = [0.28 * float(np.min(diffs))] * len(positions)

            bp = ax.boxplot(
                [all_errors[j] for j in range(len(positions))],
                positions=positions,
                widths=widths,
                manage_ticks=False,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color=color, linewidth=0.0),
                whiskerprops=dict(color=color, linewidth=0.8),
                capprops=dict(color=color, linewidth=0.8),
                boxprops=dict(color=color, linewidth=0.8),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(box_alpha)

        ax.plot(
            positions,
            medians,
            color=color,
            linewidth=2.3,
            linestyle=linestyle,
            marker="o",
            markersize=4,
            label=curve_labels[i],
        )

    ax.set_xscale(xscale)
    all_x = np.concatenate([np.asarray(diag["right_noise_levels"], dtype=float) for diag in diag_list])
    if xscale == "log":
        ax.set_xlim(all_x.min() * 0.85, all_x.max() * 1.18)
    else:
        uniq = np.unique(np.sort(all_x))
        if len(uniq) == 1:
            pad = max(0.35 * uniq[0], 1e-6)
        else:
            pad = 0.55 * np.min(np.diff(uniq))
        ax.set_xlim(max(0.0, all_x.min() - pad), all_x.max() + pad)

    if show_zero_reference:
        ax.axhline(0.0, color=baseline_color, linewidth=1.1, linestyle=":")

    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc=legend_loc)
    return ax
