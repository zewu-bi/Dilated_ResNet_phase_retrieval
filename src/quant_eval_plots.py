
import os
import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from src.prab_plot_style import PRAB_COLORS, setup_prab_style

METHOD_NN = "1D dilated resnet"
METHOD_GS = "GS"
GROUP_ORDER = ["Single", "Double"]

COLORS = {
    "target": PRAB_COLORS["target"],
    METHOD_NN: PRAB_COLORS["nn"],
    METHOD_GS: PRAB_COLORS["gs"],
}


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _finite_array(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _safe_global_idx(val_dataset, local_idx):
    if hasattr(val_dataset, "indices"):
        return int(val_dataset.indices[local_idx])
    return int(local_idx)


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


def _find_two_peak_indices(y, dx_um, min_sep_um=2.0, min_height_rel=0.15):
    y = max_normalize(y)
    if y.max() <= 0:
        return []

    min_sep_pts = max(1, int(round(min_sep_um / dx_um)))
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_height_rel:
            peaks.append(i)

    if not peaks:
        return []

    peaks = sorted(peaks, key=lambda i: y[i], reverse=True)
    chosen = []
    for p in peaks:
        if all(abs(p - q) >= min_sep_pts for q in chosen):
            chosen.append(p)
        if len(chosen) == 2:
            break
    return sorted(chosen)


def two_peak_separation(y, z, dx_um, min_sep_um=2.0, min_height_rel=0.15):
    chosen = _find_two_peak_indices(y, dx_um, min_sep_um=min_sep_um, min_height_rel=min_height_rel)
    if len(chosen) < 2:
        return np.nan
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


def _maybe_pair_from_meta(sample_meta, candidate_keys):
    if not isinstance(sample_meta, dict):
        return None
    for key in candidate_keys:
        if key in sample_meta:
            val = sample_meta[key]
            arr = np.asarray(val, dtype=float).reshape(-1)
            if arr.size >= 2:
                return arr[:2]
    return None


def estimate_double_parameters(sample_meta, target, z_centered, dx_um):
    charges = _maybe_pair_from_meta(
        sample_meta,
        ["charges", "charge_list", "q_list", "qs", "amplitudes", "weights"],
    )
    centers = _maybe_pair_from_meta(
        sample_meta,
        ["centers", "means", "mus", "mu_list", "positions", "pos", "x0", "centroids"],
    )

    q_ratio = np.nan
    separation = np.nan

    if charges is not None:
        lo = max(np.min(np.abs(charges)), 1e-12)
        hi = np.max(np.abs(charges))
        q_ratio = float(hi / lo)

    if centers is not None:
        separation = float(abs(centers[1] - centers[0]))

    if np.isfinite(separation) and np.isfinite(q_ratio):
        return separation, q_ratio

    y = l1_nonneg_normalize(target)
    peaks = _find_two_peak_indices(y, dx_um=dx_um, min_sep_um=2.0, min_height_rel=0.10)

    if len(peaks) >= 2:
        if not np.isfinite(separation):
            separation = float(abs(z_centered[peaks[1]] - z_centered[peaks[0]]))
        if not np.isfinite(q_ratio):
            mid = 0.5 * (z_centered[peaks[0]] + z_centered[peaks[1]])
            left = float(y[z_centered < mid].sum())
            right = float(y[z_centered >= mid].sum())
            lo = max(min(left, right), 1e-12)
            hi = max(left, right)
            q_ratio = float(hi / lo)

    return separation, q_ratio


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
        handles = [Patch(facecolor=colors[m], edgecolor="black", alpha=0.65, label=m) for m in methods]
        ax.legend(handles=handles, loc="best")


def _select_random_local_indices(
    val_dataset,
    tgt_files,
    target_folder,
    vis_indices,
    n_random_samples,
    sample_seed,
):
    valid_local_indices = []
    group_by_local = {}
    global_by_local = {}

    for local_idx in range(len(val_dataset)):
        global_idx = _safe_global_idx(val_dataset, local_idx)
        fname = tgt_files[global_idx]
        meta = torch.load(os.path.join(target_folder, fname), map_location="cpu", weights_only=False)
        group = infer_group(meta, fname)
        if group in ("Single", "Double"):
            valid_local_indices.append(local_idx)
            group_by_local[local_idx] = group
            global_by_local[local_idx] = global_idx

    vis_indices = tuple(vis_indices)
    forced = [idx for idx in vis_indices if idx in group_by_local]
    if n_random_samples is None:
        selected = list(valid_local_indices)
    else:
        n_random_samples = min(int(n_random_samples), len(valid_local_indices))
        remaining_pool = [idx for idx in valid_local_indices if idx not in set(forced)]
        n_extra = max(0, n_random_samples - len(forced))
        rng = np.random.default_rng(int(sample_seed))
        extra = rng.choice(remaining_pool, size=n_extra, replace=False).tolist() if n_extra > 0 else []
        selected = forced + extra

    selected = sorted(set(selected))
    selected_by_group = defaultdict(list)
    for idx in selected:
        selected_by_group[group_by_local[idx]].append(idx)

    return {
        "all": selected,
        "by_group": dict(selected_by_group),
        "group_by_local": group_by_local,
        "global_by_local": global_by_local,
        "forced_vis_indices": tuple(forced),
    }


def _run_gs_once(
    I_meas,
    gerchberg_saxton_1d_torch,
    seed,
    device,
    n_iters,
    support_mask,
    tgt_np,
    z,
    dx_um,
    fwhm_center,
    align_by_fwhm,
):
    torch.manual_seed(int(seed))
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    rho_gs = gerchberg_saxton_1d_torch(
        I_meas=I_meas,
        n_iters=n_iters,
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
    return rho_gs, (t1 - t0) * 1000.0


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
    max_per_group=None,
    n_random_samples=1000,
    sample_seed=0,
    gs_seeds=(0, 1, 2),
    gs_iters=2000,
    use_support=True,
    support_width_um=30.0,
    vis_indices=(100,),
    dense_vis_seed_count=100,
    model_name=METHOD_NN,
):
    setup_prab_style()
    model.eval()

    selection = _select_random_local_indices(
        val_dataset=val_dataset,
        tgt_files=tgt_files,
        target_folder=target_folder,
        vis_indices=vis_indices,
        n_random_samples=n_random_samples,
        sample_seed=sample_seed,
    )
    selected_local_indices = selection["all"]
    group_by_local = selection["group_by_local"]
    global_by_local = selection["global_by_local"]

    metrics = {
        "profile_nrmse": defaultdict(list),
        "spectral_error": defaultdict(list),
        "structure_error": defaultdict(list),
    }
    gs_seed_spread = defaultdict(list)
    timing = {model_name: [], METHOD_GS: []}
    sample_records = {}
    double_advantage_records = []

    vis_indices = tuple(vis_indices)
    vis_set = set(vis_indices)
    dense_vis_seeds = tuple(range(int(dense_vis_seed_count)))

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

    for local_idx in selected_local_indices:
        group = group_by_local[local_idx]
        global_idx = global_by_local[local_idx]
        fname = tgt_files[global_idx]
        sample_meta = torch.load(os.path.join(target_folder, fname), map_location="cpu", weights_only=False)

        img, tgt = val_dataset[local_idx]
        img = img.unsqueeze(0).to(device)
        measured_band = img.squeeze().detach().cpu().numpy()

        tgt_np = l1_nonneg_normalize(tgt.squeeze())
        N = len(tgt_np)
        z = np.arange(N) * dx_um
        z_centered = (np.arange(N) - N // 2) * dx_um

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
            rho_gs, gs_ms = _run_gs_once(
                I_meas=I_meas,
                gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
                seed=seed,
                device=device,
                n_iters=gs_iters,
                support_mask=support_mask,
                tgt_np=tgt_np,
                z=z,
                dx_um=dx_um,
                fwhm_center=fwhm_center,
                align_by_fwhm=align_by_fwhm,
            )
            gs_preds.append(rho_gs)
            gs_time_this_sample.append(gs_ms)

            gs_profile_err = profile_nrmse(rho_gs, tgt_np)
            gs_profile_errs.append(gs_profile_err)
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

        gs_profile_median = float(np.median(valid_gs_profile_errs)) if len(valid_gs_profile_errs) else np.nan
        gs_spec_median = float(np.median(valid_gs_spec_errs)) if len(valid_gs_spec_errs) else np.nan
        gs_struct_median = float(np.median(valid_gs_struct_errs)) if len(valid_gs_struct_errs) else np.nan

        if np.isfinite(gs_profile_median):
            metrics["profile_nrmse"][(group, METHOD_GS)].append(gs_profile_median)
            gs_seed_spread[group].append(float(np.std(valid_gs_profile_errs)))
        if np.isfinite(gs_spec_median):
            metrics["spectral_error"][(group, METHOD_GS)].append(gs_spec_median)
        if np.isfinite(gs_struct_median):
            metrics["structure_error"][(group, METHOD_GS)].append(gs_struct_median)
        timing[METHOD_GS].append(float(np.mean(gs_time_this_sample)))

        dense_gs_preds = None
        dense_q = None
        if local_idx in vis_set:
            dense_runs = []
            for seed in dense_vis_seeds:
                rho_gs, _ = _run_gs_once(
                    I_meas=I_meas,
                    gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
                    seed=seed,
                    device=device,
                    n_iters=gs_iters,
                    support_mask=support_mask,
                    tgt_np=tgt_np,
                    z=z,
                    dx_um=dx_um,
                    fwhm_center=fwhm_center,
                    align_by_fwhm=align_by_fwhm,
                )
                dense_runs.append(rho_gs)
            if len(dense_runs) > 0:
                dense_gs_preds = np.vstack(dense_runs)
                dense_q = {
                    "q05": np.percentile(dense_gs_preds, 5, axis=0),
                    "q25": np.percentile(dense_gs_preds, 25, axis=0),
                    "q50": np.percentile(dense_gs_preds, 50, axis=0),
                    "q75": np.percentile(dense_gs_preds, 75, axis=0),
                    "q95": np.percentile(dense_gs_preds, 95, axis=0),
                }

        separation, q_ratio = (np.nan, np.nan)
        if group == "Double":
            separation, q_ratio = estimate_double_parameters(sample_meta, tgt_np, z_centered, dx_um)

        record = {
            "local_idx": int(local_idx),
            "global_idx": int(global_idx),
            "fname": fname,
            "group": group,
            "target": tgt_np,
            model_name: pred_nn,
            "nn_pred": pred_nn,
            "gs_preds": [np.asarray(p, dtype=float) for p in gs_preds],
            "gs_seeds": tuple(gs_seeds),
            "gs_profile_errs": list(gs_profile_errs),
            "gs_profile_err_median": gs_profile_median,
            "gs_spec_errs": list(gs_spec_errs),
            "gs_spec_err_median": gs_spec_median,
            "gs_struct_errs": list(gs_struct_errs),
            "gs_struct_err_median": gs_struct_median,
            "nn_profile_err": nn_profile_err,
            "nn_spec_err": nn_spec_err,
            "nn_struct_err": nn_struct_err,
            "measured_band": measured_band,
            "separation_um": separation,
            "q_ratio": q_ratio,
            "selected_for_vis": local_idx in vis_set,
            "dense_gs_preds": dense_gs_preds,
            "dense_gs_quantiles": dense_q,
            "dense_vis_seed_count": int(dense_vis_seed_count) if local_idx in vis_set else 0,
        }
        sample_records[int(local_idx)] = record

        if group == "Double" and np.isfinite(separation) and np.isfinite(q_ratio) and np.isfinite(gs_profile_median):
            double_advantage_records.append({
                "local_idx": int(local_idx),
                "separation_um": float(separation),
                "q_ratio": float(q_ratio),
                "nn_error": float(nn_profile_err),
                "gs_error": float(gs_profile_median),
                "advantage": float(gs_profile_median - nn_profile_err),
            })

    examples = [sample_records[idx] for idx in vis_indices if idx in sample_records]

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
        "selected_indices": selection["by_group"],
        "selected_local_indices": list(selected_local_indices),
        "sample_records": sample_records,
        "examples": examples,
        "double_advantage_records": double_advantage_records,
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
            "forced_vis_indices_in_cache": tuple(selection["forced_vis_indices"]),
            "n_random_samples": int(len(selected_local_indices)),
            "sample_seed": int(sample_seed),
            "dense_vis_seed_count": int(dense_vis_seed_count),
            "double_heatmap_definition": {
                "x": "Peak separation (μm)",
                "y": "Charge ratio = max(Q1, Q2) / min(Q1, Q2)",
                "score": "GS median profile NRMSE - NN profile NRMSE; positive means NN is better",
            },
        },
    }


def plot_representative_examples(cache, example_idx=None, figsize=(7.2, 4.6)):
    setup_prab_style()
    cfg = cache["config"]
    model_name = cfg["model_name"]
    sample_records = cache.get("sample_records", {})

    if example_idx is None:
        vis_list = cfg.get("vis_indices", ())
        if len(vis_list) == 0:
            raise ValueError("No example_idx provided and cache['config']['vis_indices'] is empty.")
        example_idx = int(vis_list[0])

    if example_idx not in sample_records:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, f"Requested example_idx={example_idx} is not available in cache", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig, ax

    ex = sample_records[example_idx]
    target = ex["target"]
    pred_nn = ex["nn_pred"]
    N = len(target)
    z_um = (np.arange(N) - N // 2) * cfg["dx_um"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(z_um, target, color=COLORS["target"], label="Target")
    ax.plot(z_um, pred_nn, "--", color=COLORS[METHOD_NN], label=model_name)

    dense_q = ex.get("dense_gs_quantiles", None)
    if dense_q is not None:
        ax.fill_between(
            z_um,
            dense_q["q05"],
            dense_q["q95"],
            color=COLORS[METHOD_GS],
            alpha=0.18,
            label=f"GS envelope (5–95%, {ex['dense_vis_seed_count']} seeds)",
        )
    else:
        gs_preds = ex.get("gs_preds", [])
        if len(gs_preds) > 0:
            gs_stack = np.vstack(gs_preds)
            ax.fill_between(
                z_um,
                np.min(gs_stack, axis=0),
                np.max(gs_stack, axis=0),
                color=COLORS[METHOD_GS],
                alpha=0.18,
                label=f"GS envelope ({len(gs_preds)} seeds)",
            )

    ax.set_xlabel("z (μm)")
    ax.set_ylabel(r"L1-normalized $\rho(z)$")
    ax.set_title(f"Validation sample {ex['local_idx']} ({ex['group']})")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


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


def _method_boxplot(ax, values_method_1, values_method_2, positions, colors, widths=0.28):
    safe_data = [
        _finite_array(values_method_1) if len(_finite_array(values_method_1)) else np.array([np.nan]),
        _finite_array(values_method_2) if len(_finite_array(values_method_2)) else np.array([np.nan]),
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


def plot_panel_bc_double(cache, figsize=(8.8, 5.0)):
    setup_prab_style()
    model_name = cache["config"]["model_name"]

    fig, ax_left = plt.subplots(1, 1, figsize=figsize)
    ax_right = ax_left.twinx()

    x_spectral = [0.85, 1.15]
    x_sep = [1.85, 2.15]

    _method_boxplot(
        ax_left,
        cache["metrics"]["spectral_error"][("Double", model_name)],
        cache["metrics"]["spectral_error"][("Double", METHOD_GS)],
        positions=x_spectral,
        colors=[COLORS[METHOD_NN], COLORS[METHOD_GS]],
        widths=0.22,
    )
    _method_boxplot(
        ax_right,
        cache["metrics"]["structure_error"][("Double", model_name)],
        cache["metrics"]["structure_error"][("Double", METHOD_GS)],
        positions=x_sep,
        colors=[COLORS[METHOD_NN], COLORS[METHOD_GS]],
        widths=0.22,
    )

    ax_left.set_xlim(0.45, 2.55)
    ax_left.set_xticks([1.0, 2.0])
    ax_left.set_xticklabels(["Spectral mismatch", "Peak-separation error"])
    ax_left.set_ylabel(r"$\|F_{\mathrm{rec}}-F_{\mathrm{meas}}\|_2 / \|F_{\mathrm{meas}}\|_2$")
    ax_right.set_ylabel(r"$|\Delta \mathrm{separation}|$ (μm)")
    ax_left.set_title("Double-bunch spectral and separation errors")
    ax_left.grid(axis="y")

    handles = [
        Patch(facecolor=COLORS[METHOD_NN], edgecolor="black", alpha=0.65, label=model_name),
        Patch(facecolor=COLORS[METHOD_GS], edgecolor="black", alpha=0.65, label=METHOD_GS),
    ]
    ax_left.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    return fig, (ax_left, ax_right)


def plot_panel_d(cache, figsize=(7.2, 5.1)):
    setup_prab_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.text(
        0.5,
        0.5,
        "Panel (d) is no longer drawn separately.\n"
        "The GS seed dependence is summarized by the envelope in the representative example.",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def plot_double_advantage_heatmap(
    cache,
    figsize=(7.6, 5.6),
    x_bins=10,
    y_bins=10,
    min_count_per_bin=3,
    x_range=(10.0, 20.0),
    y_range=(1.0, 5.0),
):
    setup_prab_style()
    records = cache.get("double_advantage_records", [])

    xs = np.array([r["separation_um"] for r in records], dtype=float)
    ys = np.array([r["q_ratio"] for r in records], dtype=float)
    scores = np.array([r["advantage"] for r in records], dtype=float)

    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(scores)
    xs = xs[finite]
    ys = ys[finite]
    scores = scores[finite]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if len(xs) == 0:
        ax.text(0.5, 0.5, "No finite double-bunch heatmap data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig, ax

    keep = (
        (xs >= x_range[0]) & (xs <= x_range[1]) &
        (ys >= y_range[0]) & (ys <= y_range[1])
    )
    xs = xs[keep]
    ys = ys[keep]
    scores = scores[keep]

    x_edges = np.linspace(x_range[0], x_range[1], x_bins + 1)
    y_edges = np.linspace(y_range[0], y_range[1], y_bins + 1)

    sum_grid, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=scores)
    cnt_grid, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = sum_grid / cnt_grid
    mean_grid[cnt_grid < int(min_count_per_bin)] = np.nan

    mesh = ax.pcolormesh(x_edges, y_edges, mean_grid.T, shading="auto", cmap="coolwarm")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$\Delta$accuracy = NRMSE$_{GS,med}$ - NRMSE$_{NN}$")

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel("Peak separation (μm)")
    ax.set_ylabel(r"$q_{\mathrm{ratio}} = \max(Q_1,Q_2)/\min(Q_1,Q_2)$")
    ax.set_title("Double-bunch advantage map (positive: dilated ResNet better)")
    ax.grid(False)

    fig.tight_layout()
    return fig, ax


def print_cache_summary(cache):
    print("Selected validation samples")
    print(f"  total: {len(cache.get('selected_local_indices', []))}")
    print(f"  forced vis indices in cache: {cache['config'].get('forced_vis_indices_in_cache', ())}")
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

    info = cache["config"].get("double_heatmap_definition", None)
    if info is not None:
        print("\nDouble-bunch heatmap definition")
        print("-------------------------------")
        print(f"  x-axis : {info['x']}")
        print(f"  y-axis : {info['y']}")
        print(f"  score  : {info['score']}")
