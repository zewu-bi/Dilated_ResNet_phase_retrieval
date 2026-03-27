import os
import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

METHOD_NN = "1D dilated resnet"
METHOD_GS = "GS"

COLORS = {
    "target": "#1f77b4",
    METHOD_NN: "#e67e22",
    METHOD_GS: "#2ca02c",
}

GROUP_ORDER = ["Single", "Double"]


def setup_prab_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "legend.frameon": False,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "figure.dpi": 120,
    })


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _finite_array(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def l1_nonneg_normalize(y):
    y = np.clip(_to_np(y), 0, None).astype(np.float64)
    s = y.sum()
    return y / s if s > 0 else np.zeros_like(y)


def max_normalize(y):
    y = np.clip(_to_np(y), 0, None).astype(np.float64)
    m = y.max()
    return y / m if m > 0 else y


def safe_fwhm_center(y, z, fwhm_center):
    y = max_normalize(y)
    try:
        return float(fwhm_center(y, z))
    except Exception:
        s = y.sum()
        return float((z * y).sum() / s) if s > 0 else float(z[len(z) // 2])


def align_profile_to_target_center(pred, true, z, dx_um, fwhm_center, align_by_fwhm):
    c_true = safe_fwhm_center(true, z, fwhm_center)
    try:
        return _to_np(align_by_fwhm(pred, z, c_true, dx_um))
    except Exception:
        return _to_np(pred)


def profile_nrmse(pred, true):
    pred = _to_np(pred)
    true = _to_np(true)
    denom = np.linalg.norm(true)
    return np.linalg.norm(pred - true) / denom if denom > 0 else np.nan


def compute_band_spectral_error(profile, measured_band, dx_um, f_min, f_max, pad_factor,
                                compute_form_factor, k_to_THz):
    profile = l1_nonneg_normalize(profile)
    measured_band = np.clip(_to_np(measured_band), 0, None).astype(np.float64)

    z = np.arange(len(profile)) * dx_um
    k_rec, F2_rec = compute_form_factor(z, profile, pad_factor=pad_factor)
    freq_rec = k_to_THz(k_rec)

    mask = (freq_rec >= f_min) & (freq_rec <= f_max)
    if not np.any(mask):
        return np.nan

    freq_band = freq_rec[mask]
    F2_band = np.clip(F2_rec[mask], 0, None)
    freq_meas = np.linspace(f_min, f_max, len(measured_band))
    F2_interp = np.interp(freq_meas, freq_band, F2_band)

    m1 = measured_band.max()
    m2 = F2_interp.max()
    measured_band = measured_band / (m1 if m1 > 0 else 1.0)
    F2_interp = F2_interp / (m2 if m2 > 0 else 1.0)

    denom = np.linalg.norm(measured_band)
    return np.linalg.norm(F2_interp - measured_band) / denom if denom > 0 else np.nan


def fwhm_width(y, z):
    y = max_normalize(y)
    if y.max() <= 0:
        return np.nan
    idx = np.where(y >= 0.5)[0]
    return float(z[idx[-1]] - z[idx[0]]) if len(idx) >= 2 else np.nan


def two_peak_separation(y, z, dx_um, min_sep_um=2.0, min_height_rel=0.15):
    y = max_normalize(y)
    if y.max() <= 0:
        return np.nan

    min_sep_pts = max(1, int(round(min_sep_um / dx_um)))
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_height_rel:
            peaks.append(i)
    if not peaks:
        return np.nan

    peaks = sorted(peaks, key=lambda i: y[i], reverse=True)
    chosen = []
    for p in peaks:
        if all(abs(p - q) >= min_sep_pts for q in chosen):
            chosen.append(p)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return np.nan
    chosen = sorted(chosen)
    return float(abs(z[chosen[1]] - z[chosen[0]]))


def infer_group(sample_meta, fname):
    if isinstance(sample_meta, dict):
        if "charges" in sample_meta:
            n = len(sample_meta["charges"])
            if n == 1:
                return "Single"
            if n == 2:
                return "Double"
        if "shapes" in sample_meta:
            n = len(sample_meta["shapes"])
            if n == 1:
                return "Single"
            if n == 2:
                return "Double"

    stem = os.path.basename(fname)
    if re.search(r"(^|[_-])G1([_-]|\.|$)", stem):
        return "Single"
    if re.search(r"(^|[_-])G2([_-]|\.|$)", stem):
        return "Double"
    return None


def make_support_mask(N, dx_um, width_um, device):
    z_um = (np.arange(N) - N // 2) * dx_um
    mask_np = (np.abs(z_um) <= width_um / 2).astype(np.float32)
    return torch.tensor(mask_np, dtype=torch.float32, device=device)


def _grouped_boxplot(ax, data_by_key, groups, methods, ylabel, title, xticklabels, colors, show_legend=False):
    positions, data, facecolors = [], [], []
    base_positions = np.arange(len(groups)) * 3.0
    offsets = np.linspace(-0.45, 0.45, len(methods))

    for gi, group in enumerate(groups):
        for mi, method in enumerate(methods):
            vals = _finite_array(data_by_key[(group, method)])
            positions.append(base_positions[gi] + offsets[mi])
            data.append(vals)
            facecolors.append(colors[method])

    if not any(len(v) > 0 for v in data):
        ax.text(0.5, 0.5, "No finite data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # matplotlib boxplot cannot handle empty arrays reliably; replace empty ones with [nan]
    safe_data = [v if len(v) > 0 else np.array([np.nan]) for v in data]

    bp = ax.boxplot(
        safe_data,
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
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y")

    if show_legend:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=colors[m], edgecolor="black", alpha=0.65, label=m) for m in methods]
        ax.legend(handles=handles, loc="best")


def build_quantitative_cache(
    model,
    val_dataset,
    tgt_files,
    target_folder,
    device,
    dx_um,
    f_min,
    f_max,
    f_display_max,
    pad_factor,
    compute_form_factor,
    k_to_THz,
    forward_spectrum_fft,
    gerchberg_saxton_1d_torch,
    fwhm_center,
    align_by_fwhm,
    max_per_group=500,
    gs_seeds=(0, 1, 2, 3, 4),
    gs_iters=2000,
    use_support=True,
    support_width_um=30.0,
    vis_indices=(101, 102, 103),
    model_name=METHOD_NN,
):
    setup_prab_style()
    model.eval()

    selected_indices = defaultdict(list)
    for local_idx, global_idx in enumerate(val_dataset.indices):
        fname = tgt_files[global_idx]
        meta = torch.load(os.path.join(target_folder, fname), map_location="cpu", weights_only=False)
        group = infer_group(meta, fname)
        if group in ("Single", "Double") and len(selected_indices[group]) < max_per_group:
            selected_indices[group].append(local_idx)

    metrics = {
        "profile_nrmse": defaultdict(list),
        "spectral_error": defaultdict(list),
        "structure_error": defaultdict(list),
    }
    gs_seed_spread = defaultdict(list)
    timing = {model_name: [], METHOD_GS: []}
    example_bank = []
    examples = []
    vis_indices = tuple(vis_indices)
    vis_set = set(vis_indices)

    def run_nn(img):
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(img)
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return pred, (t1 - t0) * 1000.0

    for group in GROUP_ORDER:
        for local_idx in selected_indices[group]:
            img, tgt = val_dataset[local_idx]
            img = img.unsqueeze(0).to(device)
            measured_band = img.squeeze().detach().cpu().numpy()

            tgt_np = l1_nonneg_normalize(tgt.squeeze())
            N = len(tgt_np)
            z = np.arange(N) * dx_um

            pred_nn_t, nn_ms = run_nn(img)
            pred_nn = l1_nonneg_normalize(pred_nn_t.squeeze().cpu().numpy())
            pred_nn = l1_nonneg_normalize(
                align_profile_to_target_center(pred_nn, tgt_np, z, dx_um, fwhm_center, align_by_fwhm)
            )

            nn_profile_err = profile_nrmse(pred_nn, tgt_np)
            nn_spec_err = compute_band_spectral_error(
                pred_nn, measured_band, dx_um, f_min, f_max, pad_factor, compute_form_factor, k_to_THz
            )
            if group == "Single":
                true_struct = fwhm_width(tgt_np, z)
                nn_struct = fwhm_width(pred_nn, z)
            else:
                true_struct = two_peak_separation(tgt_np, z, dx_um)
                nn_struct = two_peak_separation(pred_nn, z, dx_um)
            nn_struct_err = np.nan if np.isnan(true_struct) or np.isnan(nn_struct) else abs(nn_struct - true_struct)

            metrics["profile_nrmse"][(group, model_name)].append(nn_profile_err)
            metrics["spectral_error"][(group, model_name)].append(nn_spec_err)
            if np.isfinite(nn_struct_err):
                metrics["structure_error"][(group, model_name)].append(nn_struct_err)
            timing[model_name].append(nn_ms)

            rho_true = torch.tensor(tgt_np, dtype=torch.float32, device=device)
            I_meas, _ = forward_spectrum_fft(rho_true)
            I_meas = I_meas.real
            I_meas = I_meas / torch.clamp(torch.max(I_meas), min=1e-12)

            support_mask = make_support_mask(N, dx_um, support_width_um, device) if use_support else None

            gs_profile_errs = []
            gs_spec_errs = []
            gs_struct_errs = []
            gs_preds = []
            gs_time_this_sample = []

            for seed in gs_seeds:
                torch.manual_seed(int(seed))
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                rho_gs = gerchberg_saxton_1d_torch(
                    I_meas=I_meas,
                    n_iters=gs_iters,
                    support_mask=support_mask,
                    smooth=True,
                    device=device,
                )
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                rho_gs = l1_nonneg_normalize(rho_gs.detach().cpu().numpy())
                rho_gs = l1_nonneg_normalize(
                    align_profile_to_target_center(rho_gs, tgt_np, z, dx_um, fwhm_center, align_by_fwhm)
                )

                gs_preds.append(rho_gs)
                gs_time_this_sample.append((t1 - t0) * 1000.0)

                gs_profile_errs.append(profile_nrmse(rho_gs, tgt_np))
                gs_spec_errs.append(
                    compute_band_spectral_error(
                        rho_gs, measured_band, dx_um, f_min, f_max, pad_factor, compute_form_factor, k_to_THz
                    )
                )
                if group == "Single":
                    gs_struct = fwhm_width(rho_gs, z)
                else:
                    gs_struct = two_peak_separation(rho_gs, z, dx_um)
                gs_struct_err = np.nan if np.isnan(true_struct) or np.isnan(gs_struct) else abs(gs_struct - true_struct)
                gs_struct_errs.append(gs_struct_err)

            valid_gs_profile_errs = _finite_array(gs_profile_errs)
            valid_gs_spec_errs = _finite_array(gs_spec_errs)
            valid_gs_struct_errs = _finite_array(gs_struct_errs)

            if len(valid_gs_profile_errs) > 0:
                metrics["profile_nrmse"][(group, METHOD_GS)].append(float(np.median(valid_gs_profile_errs)))
                gs_seed_spread[group].append(float(np.std(valid_gs_profile_errs)))
            if len(valid_gs_spec_errs) > 0:
                metrics["spectral_error"][(group, METHOD_GS)].append(float(np.median(valid_gs_spec_errs)))
            if len(valid_gs_struct_errs) > 0:
                metrics["structure_error"][(group, METHOD_GS)].append(float(np.median(valid_gs_struct_errs)))
            timing[METHOD_GS].append(float(np.mean(gs_time_this_sample)))

            example_record = {
                "local_idx": local_idx,
                "group": group,
                "target": tgt_np,
                model_name: pred_nn,
                METHOD_GS: gs_preds[0] if len(gs_preds) else None,
                "measured_band": measured_band,
            }
            example_bank.append(example_record)
            if local_idx in vis_set:
                ex = dict(example_record)
                ex["vis_pos"] = list(vis_indices).index(local_idx)
                examples.append(ex)

    # Fallback: if requested vis_indices are not in selected single/double pool, fill from available examples.
    examples = sorted(examples, key=lambda d: d["vis_pos"])
    if len(examples) < len(vis_indices):
        used = {ex["local_idx"] for ex in examples}
        for ex in example_bank:
            if ex["local_idx"] not in used:
                examples.append(ex)
                used.add(ex["local_idx"])
            if len(examples) >= len(vis_indices):
                break
    examples = examples[: len(vis_indices)] if len(vis_indices) > 0 else examples[:3]

    summary = {}
    for metric_name, metric_dict in metrics.items():
        summary[metric_name] = {}
        for group in GROUP_ORDER:
            for method in [model_name, METHOD_GS]:
                vals = _finite_array(metric_dict[(group, method)])
                summary[metric_name][(group, method)] = {
                    "n": int(len(vals)),
                    "median": float(np.median(vals)) if len(vals) else np.nan,
                    "q25": float(np.percentile(vals, 25)) if len(vals) else np.nan,
                    "q75": float(np.percentile(vals, 75)) if len(vals) else np.nan,
                    "mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "std": float(np.std(vals)) if len(vals) else np.nan,
                }

    timing_summary = {}
    for method in [model_name, METHOD_GS]:
        vals = _finite_array(timing[method])
        timing_summary[method] = {
            "mean_ms": float(np.mean(vals)) if len(vals) else np.nan,
            "std_ms": float(np.std(vals)) if len(vals) else np.nan,
            "n": int(len(vals)),
        }

    return {
        "metrics": metrics,
        "gs_seed_spread": gs_seed_spread,
        "selected_indices": dict(selected_indices),
        "examples": examples,
        "summary": summary,
        "timing": timing_summary,
        "config": {
            "dx_um": dx_um,
            "f_min": f_min,
            "f_max": f_max,
            "f_display_max": f_display_max,
            "pad_factor": pad_factor,
            "model_name": model_name,
            "gs_iters": gs_iters,
            "gs_seeds": tuple(gs_seeds),
            "use_support": use_support,
            "support_width_um": support_width_um,
            "vis_indices": tuple(vis_indices),
        },
    }


def plot_representative_examples(cache, figsize=(8.5, 10.5)):
    setup_prab_style()
    cfg = cache["config"]
    model_name = cfg["model_name"]
    examples = cache.get("examples", [])

    if len(examples) == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No representative single/double examples available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig, [ax]

    nrows = len(examples)
    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=False)
    if nrows == 1:
        axes = [axes]

    for ax, ex in zip(axes, examples):
        target = ex["target"]
        pred_nn = ex[model_name]
        pred_gs = ex[METHOD_GS]
        N = len(target)
        z_um = (np.arange(N) - N // 2) * cfg["dx_um"]

        ax.plot(z_um, target, color=COLORS["target"], label="Target")
        ax.plot(z_um, pred_nn, "--", color=COLORS[METHOD_NN], label=model_name)
        gs_label = f"GS ({cfg['gs_iters']} iters"
        if cfg["use_support"]:
            gs_label += f", support={cfg['support_width_um']:.0f} μm)"
        else:
            gs_label += ")"
        ax.plot(z_um, pred_gs, "-.", color=COLORS[METHOD_GS], label=gs_label)
        ax.set_xlabel("z (μm)")
        ax.set_ylabel(r"L1-normalized $\rho(z)$")
        ax.set_title(f"Validation sample {ex['local_idx']} ({ex['group']})")
        ax.grid(True)
        ax.legend(loc="best")

    fig.tight_layout()
    return fig, axes


def plot_panel_a(cache, figsize=(7.2, 5.1)):
    setup_prab_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    model_name = cache["config"]["model_name"]
    _grouped_boxplot(
        ax,
        cache["metrics"]["profile_nrmse"],
        groups=GROUP_ORDER,
        methods=[model_name, METHOD_GS],
        ylabel="Profile NRMSE",
        title="(a) Profile reconstruction error",
        xticklabels=["Single", "Double"],
        colors={model_name: COLORS[METHOD_NN], METHOD_GS: COLORS[METHOD_GS]},
        show_legend=True,
    )
    fig.tight_layout()
    return fig, ax


def plot_panel_bc_double(cache, figsize=(11.0, 4.8)):
    setup_prab_style()
    model_name = cache["config"]["model_name"]
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    _grouped_boxplot(
        axes[0],
        cache["metrics"]["spectral_error"],
        groups=["Double"],
        methods=[model_name, METHOD_GS],
        ylabel="||F_rec - F_meas||_2 / ||F_meas||_2",
        title="(b) Double-bunch spectral mismatch",
        xticklabels=["Double"],
        colors={model_name: COLORS[METHOD_NN], METHOD_GS: COLORS[METHOD_GS]},
        show_legend=True,
    )

    _grouped_boxplot(
        axes[1],
        cache["metrics"]["structure_error"],
        groups=["Double"],
        methods=[model_name, METHOD_GS],
        ylabel=r"$|\Delta$ peak separation$|$ (μm)",
        title="(c) Double-bunch separation error",
        xticklabels=["Double"],
        colors={model_name: COLORS[METHOD_NN], METHOD_GS: COLORS[METHOD_GS]},
        show_legend=True,
    )

    fig.tight_layout()
    return fig, axes


def plot_panel_d(cache, figsize=(7.2, 5.1)):
    setup_prab_style()
    model_name = cache["config"]["model_name"]
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    single = _finite_array(cache["gs_seed_spread"].get("Single", []))
    double = _finite_array(cache["gs_seed_spread"].get("Double", []))

    if len(single) == 0 and len(double) == 0:
        ax.text(0.5, 0.5, "No finite GS seed-spread values available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig, ax

    data = [single if len(single) else np.array([np.nan]), double if len(double) else np.array([np.nan])]
    positions = [0, 3]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.9,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS[METHOD_GS])
        patch.set_alpha(0.65)

    ax.axhline(0.0, color=COLORS[METHOD_NN], linestyle="--", linewidth=2.0, label=f"{model_name} (deterministic)")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Single", "Double"])
    ax.set_ylabel("Std. of GS profile NRMSE across seeds")
    ax.set_title("(d) GS procedural variability")
    ax.grid(axis="y")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def print_cache_summary(cache):
    print("Selected validation samples")
    for group in GROUP_ORDER:
        print(f"  {group}: {len(cache['selected_indices'].get(group, []))}")

    print("\nTiming summary")
    print("--------------")
    for method, stats in cache["timing"].items():
        print(f"{method:>18s}: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms  (n={stats['n']})")

    print("\nMetric summary (median [Q1, Q3])")
    print("---------------------------------")
    for metric_name, block in cache["summary"].items():
        print(f"\n[{metric_name}]")
        for group in GROUP_ORDER:
            for method in [cache["config"]["model_name"], METHOD_GS]:
                s = block[(group, method)]
                print(
                    f"{group:>6s} | {method:>18s} : "
                    f"{s['median']:.4f} [{s['q25']:.4f}, {s['q75']:.4f}]  n={s['n']}"
                )
