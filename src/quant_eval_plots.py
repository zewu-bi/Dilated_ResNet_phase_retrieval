
import os
import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
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


def _weighted_sigma(z, y):
    y = np.clip(np.asarray(y, dtype=float), 0, None)
    s = y.sum()
    if s <= 0:
        return np.nan
    mu = float((z * y).sum() / s)
    var = float(((z - mu) ** 2 * y).sum() / s)
    return np.sqrt(max(var, 0.0))


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



def estimate_double_sigmas(sample_meta, target, z_centered, dx_um):
    centers = _maybe_pair_from_meta(
        sample_meta,
        ["centers", "means", "mus", "mu_list", "positions", "pos", "x0", "centroids"],
    )
    sigmas = _maybe_pair_from_meta(
        sample_meta,
        ["sigmas", "sigma_list", "widths", "width_list", "stds", "std_list", "sigma"],
    )

    if sigmas is not None and centers is not None:
        order = np.argsort(np.asarray(centers, dtype=float))
        sigmas = np.asarray(sigmas, dtype=float)[order]
        return float(sigmas[0]), float(sigmas[1])
    if sigmas is not None:
        sigmas = np.asarray(sigmas, dtype=float)
        return float(sigmas[0]), float(sigmas[1])

    y = l1_nonneg_normalize(target)
    peaks = _find_two_peak_indices(y, dx_um=dx_um, min_sep_um=2.0, min_height_rel=0.10)
    if len(peaks) < 2:
        return np.nan, np.nan

    valley_rel = np.argmin(y[peaks[0]: peaks[1] + 1])
    valley = int(peaks[0] + valley_rel)

    left_mask = np.zeros_like(y, dtype=bool)
    right_mask = np.zeros_like(y, dtype=bool)
    left_mask[: valley + 1] = True
    right_mask[valley + 1:] = True

    sigma_left = _weighted_sigma(z_centered[left_mask], y[left_mask])
    sigma_right = _weighted_sigma(z_centered[right_mask], y[right_mask])
    return float(sigma_left), float(sigma_right)


def _grouped_boxplot(ax, data_by_key, groups, methods, ylabel, title, xticklabels, colors, show_legend=False, legend_loc="upper right"):
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
        ax.legend(handles=handles, loc=legend_loc)


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
        sigma_left, sigma_right = (np.nan, np.nan)
        if group == "Double":
            separation, q_ratio = estimate_double_parameters(sample_meta, tgt_np, z_centered, dx_um)
            sigma_left, sigma_right = estimate_double_sigmas(sample_meta, tgt_np, z_centered, dx_um)

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
            "sigma_left_um": sigma_left,
            "sigma_right_um": sigma_right,
            "selected_for_vis": local_idx in vis_set,
            "dense_gs_preds": dense_gs_preds,
            "dense_gs_quantiles": dense_q,
            "dense_vis_seed_count": int(dense_vis_seed_count) if local_idx in vis_set else 0,
        }
        sample_records[int(local_idx)] = record

        if group == "Double" and np.isfinite(sigma_left) and np.isfinite(sigma_right) and np.isfinite(gs_profile_median):
            double_advantage_records.append({
                "local_idx": int(local_idx),
                "sigma_left_um": float(sigma_left),
                "sigma_right_um": float(sigma_right),
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
                "x": "Left-bunch effective sigma (μm)",
                "y": "Right-bunch effective sigma (μm)",
                "score": "GS median profile NRMSE - NN profile NRMSE; positive means NN is better",
            },
        },
    }


def plot_representative_examples(
    cache,
    example_idx=None,
    figsize=(7.2, 4.6),
    title=None,
    xlabel="z (μm)",
    ylabel=r"L1-normalized $\rho(z)$",
    target_label="Target",
    nn_label=None,
    gs_label=None,
    legend_loc="upper right",
):
    setup_prab_style()
    cfg = cache["config"]
    model_name = cfg["model_name"]
    sample_records = cache.get("sample_records", {})

    if nn_label is None:
        nn_label = model_name

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
    ax.plot(z_um, target, color=COLORS["target"], label=target_label)
    ax.plot(z_um, pred_nn, "--", color=COLORS[METHOD_NN], label=nn_label)

    dense_q = ex.get("dense_gs_quantiles", None)
    if dense_q is not None:
        if gs_label is None:
            gs_label = f"GS envelope (5–95%, {ex['dense_vis_seed_count']} seeds)"
        ax.fill_between(
            z_um,
            dense_q["q05"],
            dense_q["q95"],
            color=COLORS[METHOD_GS],
            alpha=0.18,
            label=gs_label,
        )
    else:
        gs_preds = ex.get("gs_preds", [])
        if len(gs_preds) > 0:
            if gs_label is None:
                gs_label = f"GS envelope ({len(gs_preds)} seeds)"
            gs_stack = np.vstack(gs_preds)
            ax.fill_between(
                z_um,
                np.min(gs_stack, axis=0),
                np.max(gs_stack, axis=0),
                color=COLORS[METHOD_GS],
                alpha=0.18,
                label=gs_label,
            )

    if title is None:
        title = f"Validation sample {ex['local_idx']} ({ex['group']})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc=legend_loc)
    fig.tight_layout()
    return fig, ax


def plot_panel_a(
    cache,
    figsize=(7.2, 5.1),
    title="(a) Profile reconstruction error",
    ylabel="Profile NRMSE",
    xticklabels=("Single", "Double"),
    legend_loc="upper right",
):
    setup_prab_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    model_name = cache["config"]["model_name"]
    _grouped_boxplot(
        ax,
        cache["metrics"]["profile_nrmse"],
        groups=GROUP_ORDER,
        methods=[model_name, METHOD_GS],
        ylabel=ylabel,
        title=title,
        xticklabels=list(xticklabels),
        colors={model_name: COLORS[METHOD_NN], METHOD_GS: COLORS[METHOD_GS]},
        show_legend=True,
        legend_loc=legend_loc,
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


def plot_panel_bc_double(
    cache,
    figsize=(8.8, 5.0),
    title="Double-bunch spectral and separation errors",
    left_ylabel=r"$\|F_{\mathrm{rec}}-F_{\mathrm{meas}}\|_2 / \|F_{\mathrm{meas}}\|_2$",
    right_ylabel=r"$|\Delta \mathrm{separation}|$ (μm)",
    xticklabels=("Spectral mismatch", "Peak-separation error"),
    legend_loc="upper right",
):
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
    ax_left.set_xticklabels(list(xticklabels))
    ax_left.set_ylabel(left_ylabel)
    ax_right.set_ylabel(right_ylabel)
    ax_left.set_title(title)
    ax_left.grid(axis="y")

    handles = [
        Patch(facecolor=COLORS[METHOD_NN], edgecolor="black", alpha=0.65, label=model_name),
        Patch(facecolor=COLORS[METHOD_GS], edgecolor="black", alpha=0.65, label=METHOD_GS),
    ]
    ax_left.legend(handles=handles, loc=legend_loc)
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
    x_range=None,
    y_range=None,
    title="Double-bunch advantage map (positive: dilated ResNet better)",
    xlabel=r"Left-bunch effective $\sigma$ (μm)",
    ylabel=r"Right-bunch effective $\sigma$ (μm)",
    cbar_label=r"$\Delta$accuracy = NRMSE$_{GS,med}$ - NRMSE$_{NN}$",
):
    setup_prab_style()
    records = cache.get("double_advantage_records", [])

    xs = np.array([r["sigma_left_um"] for r in records], dtype=float)
    ys = np.array([r["sigma_right_um"] for r in records], dtype=float)
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
        return fig, ax, None

    if x_range is None:
        x_pad = 0.05 * max(xs.max() - xs.min(), 1e-6)
        x_range = (xs.min() - x_pad, xs.max() + x_pad)
    if y_range is None:
        y_pad = 0.05 * max(ys.max() - ys.min(), 1e-6)
        y_range = (ys.min() - y_pad, ys.max() + y_pad)

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

    finite_grid = mean_grid[np.isfinite(mean_grid)]
    vmax = float(np.max(np.abs(finite_grid))) if finite_grid.size else 1.0
    vmax = max(vmax, 1e-12)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    mesh = ax.pcolormesh(x_edges, y_edges, mean_grid.T, shading="auto", cmap="bwr", norm=norm)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

    fig.tight_layout()
    return fig, ax, cbar




def _set_torch_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _normalize_profile(y):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0.0, None)
    s = y.sum()
    if s <= 0:
        return np.zeros_like(y)
    return y / s


def _weighted_relative_error(pred, ref, weight):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    weight = np.asarray(weight, dtype=float)
    num = np.sqrt(np.sum(weight * (pred - ref) ** 2))
    den = np.sqrt(np.sum(weight * ref ** 2))
    return num / den if den > 0 else np.nan


def _maybe_align_profile(pred, z, c_ref, dx_um, align_by_fwhm):
    try:
        return np.asarray(align_by_fwhm(pred, z, c_ref, dx_um), dtype=float)
    except Exception:
        return np.asarray(pred, dtype=float)


def _reconstruct_nn_from_band(model, img_band):
    with torch.no_grad():
        pred = model(img_band).squeeze().detach().cpu().numpy()
    return _normalize_profile(pred)


def _reconstruct_gs_from_profile(
    tgt_profile,
    nl,
    noise_seed,
    gs_seed,
    device,
    dx_um,
    forward_spectrum_fft,
    gerchberg_saxton_1d_torch,
    gs_iters,
    use_support,
    support_width_um,
):
    rho_true = torch.tensor(tgt_profile, dtype=torch.float32, device=device)
    I_meas, _ = forward_spectrum_fft(rho_true)
    I_meas = I_meas.real
    I_meas = I_meas / torch.clamp(torch.max(I_meas), min=1e-12)

    _set_torch_seed(noise_seed)
    noise = nl * torch.randn_like(I_meas)
    I_noisy = torch.clamp(I_meas + noise, min=0.0)
    I_noisy = I_noisy / torch.clamp(torch.max(I_noisy), min=1e-12)

    support_mask = None
    if use_support:
        support_mask = make_support_mask(len(tgt_profile), dx_um, support_width_um, device)

    _set_torch_seed(gs_seed)
    rho_gs = gerchberg_saxton_1d_torch(
        I_meas=I_noisy,
        n_iters=gs_iters,
        support_mask=support_mask,
        smooth=True,
        device=device,
    )
    return _normalize_profile(rho_gs.detach().cpu().numpy())


def _force_monotone_by_trial(error_curves_2d):
    arr = np.asarray(error_curves_2d, dtype=float).copy()
    for j in range(arr.shape[1]):
        arr[:, j] = np.maximum.accumulate(arr[:, j])
    return arr



def _compute_noise_robustness_diagnostics(
    method,
    idx,
    val_dataset,
    device,
    dx_um,
    fwhm_center,
    align_by_fwhm,
    model=None,
    forward_spectrum_fft=None,
    gerchberg_saxton_1d_torch=None,
    rep_levels=(1e-4, 1e-3, 1e-2),
    noise_levels=None,
    right_noise_levels=None,
    n_trials=50,
    gs_seed_fixed=0,
    gs_iters=2000,
    use_support=True,
    support_width_um=30.0,
    enforce_monotone_display=False,
):
    if right_noise_levels is None:
        right_noise_levels = noise_levels
    if right_noise_levels is None:
        right_noise_levels = np.logspace(-4, -2, 15)

    right_noise_levels = np.asarray(right_noise_levels, dtype=float)
    rep_levels = tuple(float(x) for x in rep_levels)

    if np.any(right_noise_levels <= 0):
        raise ValueError('right_noise_levels must be strictly positive so the right panel can use a numeric axis.')

    img, tgt = val_dataset[int(idx)]
    img = img.unsqueeze(0).to(device)
    tgt = tgt.squeeze().detach().cpu().numpy()
    z = np.arange(len(tgt)) * dx_um
    tgt_norm = _normalize_profile(tgt)

    if method == 'nn':
        if model is None:
            raise ValueError('model must be provided for method="nn".')
        model.eval()
        base_profile = _reconstruct_nn_from_band(model, img)
        method_label = METHOD_NN
    elif method == 'gs':
        if forward_spectrum_fft is None or gerchberg_saxton_1d_torch is None:
            raise ValueError('forward_spectrum_fft and gerchberg_saxton_1d_torch must be provided for method="gs".')
        base_profile = _reconstruct_gs_from_profile(
            tgt_profile=tgt_norm,
            nl=0.0,
            noise_seed=0,
            gs_seed=gs_seed_fixed,
            device=device,
            dx_um=dx_um,
            forward_spectrum_fft=forward_spectrum_fft,
            gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
            gs_iters=gs_iters,
            use_support=use_support,
            support_width_um=support_width_um,
        )
        method_label = METHOD_GS
    else:
        raise ValueError("method must be 'nn' or 'gs'.")

    base_max = base_profile / max(np.max(base_profile), 1e-12)
    c_ref = safe_fwhm_center(base_max, z, fwhm_center)
    base_aligned = _maybe_align_profile(base_profile, z, c_ref, dx_um, align_by_fwhm)
    weight = base_aligned / max(np.max(base_aligned), 1e-12)

    rep_profiles = []
    for nl in rep_levels:
        if method == 'nn':
            _set_torch_seed(0)
            noise = nl * torch.randn_like(img)
            img_noisy = torch.clamp(img + noise, min=0.0)
            pred = _reconstruct_nn_from_band(model, img_noisy)
        else:
            pred = _reconstruct_gs_from_profile(
                tgt_profile=tgt_norm,
                nl=nl,
                noise_seed=0,
                gs_seed=gs_seed_fixed,
                device=device,
                dx_um=dx_um,
                forward_spectrum_fft=forward_spectrum_fft,
                gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
                gs_iters=gs_iters,
                use_support=use_support,
                support_width_um=support_width_um,
            )
        pred = _maybe_align_profile(pred, z, c_ref, dx_um, align_by_fwhm)
        rep_profiles.append((float(nl), pred))

    all_errors = []
    for nl in right_noise_levels:
        trial_errors = []
        for seed in range(int(n_trials)):
            if method == 'nn':
                _set_torch_seed(seed)
                noise = nl * torch.randn_like(img)
                img_noisy = torch.clamp(img + noise, min=0.0)
                pred = _reconstruct_nn_from_band(model, img_noisy)
                pred = _maybe_align_profile(pred, z, c_ref, dx_um, align_by_fwhm)
            else:
                pred = _reconstruct_gs_from_profile(
                    tgt_profile=tgt_norm,
                    nl=nl,
                    noise_seed=seed,
                    gs_seed=gs_seed_fixed,
                    device=device,
                    dx_um=dx_um,
                    forward_spectrum_fft=forward_spectrum_fft,
                    gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
                    gs_iters=gs_iters,
                    use_support=use_support,
                    support_width_um=support_width_um,
                )
                pred = _maybe_align_profile(pred, z, c_ref, dx_um, align_by_fwhm)
            trial_errors.append(_weighted_relative_error(pred, base_aligned, weight))
        all_errors.append(trial_errors)

    all_errors = np.asarray(all_errors, dtype=float)
    raw_errors = all_errors.copy()
    if enforce_monotone_display:
        all_errors = _force_monotone_by_trial(all_errors)
    medians = np.median(all_errors, axis=1)

    return {
        'idx': int(idx),
        'method': method,
        'method_label': method_label,
        'z_um': z,
        'target': tgt_norm,
        'baseline_profile': base_aligned,
        'rep_profiles': rep_profiles,
        'right_noise_levels': right_noise_levels.astype(float),
        'all_errors_displayed': all_errors,
        'all_errors_raw': raw_errors,
        'median_displayed': medians,
        'n_trials': int(n_trials),
        'enforce_monotone_display': bool(enforce_monotone_display),
        'gs_seed_fixed': int(gs_seed_fixed),
        'gs_iters': int(gs_iters),
        'use_support': bool(use_support),
        'support_width_um': float(support_width_um),
    }


def _plot_noise_robustness_left_from_diag(
    diag,
    figsize=(6.4, 5.0),
    title=None,
    xlabel_profile='z (μm)',
    left_ylabel='Normalized charge',
    baseline_label='0 noise',
    rep_labels=None,
    legend_loc='upper right',
    baseline_color='black',
    rep_colors=None,
    rep_linestyles=None,
    caption=None,
):
    setup_prab_style()

    rep_profiles = diag['rep_profiles']
    if rep_colors is None:
        rep_colors = ['#1f77b4', '#e67e22', '#d62728']
    if rep_linestyles is None:
        rep_linestyles = ['--', '-.', ':']
    if rep_labels is None:
        rep_labels = [f'{nl*100:.2f}% noise' for nl, _ in rep_profiles]
    if len(rep_labels) != len(rep_profiles):
        raise ValueError('rep_labels must have the same length as diag["rep_profiles"].')

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(diag['z_um'], diag['baseline_profile'], color=baseline_color, linestyle='-', linewidth=2.2, label=baseline_label)
    for i, ((_, prof), label) in enumerate(zip(rep_profiles, rep_labels)):
        ax.plot(
            diag['z_um'], prof,
            color=rep_colors[i % len(rep_colors)],
            linestyle=rep_linestyles[i % len(rep_linestyles)],
            linewidth=2.0, label=label,
        )
    ax.set_xlabel(xlabel_profile)
    ax.set_ylabel(left_ylabel)
    ax.set_title(title or '')
    ax.grid(alpha=0.25)
    ax.legend(loc=legend_loc)
    if caption is not None:
        fig.text(0.5, 0.02, caption, ha='center', va='center', fontsize=11)
    fig.tight_layout()
    fig._qep_noise_robustness = diag
    return fig, ax


def _plot_noise_robustness_right_from_diag(
    diag,
    figsize=(6.8, 5.0),
    title=None,
    xlabel_noise='Noise level',
    right_ylabel='Peak-weighted relative error to zero-noise output',
    legend_loc='upper right',
    caption=None,
    box_facecolor=None,
    median_color='black',
    baseline_color='black',
    right_xscale='log',
    right_xticks=None,
    right_xticklabels=None,
    curve_label='Median trend',
    zero_ref_label='Zero-noise reference',
):
    setup_prab_style()

    positions = np.asarray(diag['right_noise_levels'], dtype=float)
    all_errors = np.asarray(diag['all_errors_displayed'], dtype=float)
    medians = np.asarray(diag['median_displayed'], dtype=float)

    if np.any(positions <= 0):
        raise ValueError('diag["right_noise_levels"] must be strictly positive.')
    if right_xscale not in ('log', 'linear'):
        raise ValueError("right_xscale must be 'log' or 'linear'.")

    if box_facecolor is None:
        box_facecolor = COLORS[diag['method_label']]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if len(positions) == 1:
        widths = [0.25 * positions[0]] if right_xscale == 'log' else [max(0.25 * positions[0], 1e-6)]
    elif right_xscale == 'log':
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
    for patch in bp['boxes']:
        patch.set_facecolor(box_facecolor)
        patch.set_alpha(0.45)

    ax.plot(positions, medians, color=median_color, linewidth=2.2, marker='o', markersize=4, label=curve_label)
    ax.axhline(0.0, color=baseline_color, linewidth=1.2, linestyle=':', label=zero_ref_label)

    ax.set_xscale(right_xscale)
    if right_xscale == 'log':
        ax.set_xlim(positions.min() * 0.85, positions.max() * 1.18)
    else:
        if len(positions) == 1:
            pad = max(0.35 * positions[0], 1e-6)
        else:
            pad = 0.55 * np.min(np.diff(np.sort(positions)))
        ax.set_xlim(max(0.0, positions.min() - pad), positions.max() + pad)

    if right_xticks is not None:
        ax.set_xticks(right_xticks)
    if right_xticklabels is not None:
        ax.set_xticklabels(right_xticklabels)

    ax.set_xlabel(xlabel_noise)
    ax.set_ylabel(right_ylabel)
    ax.set_title(title or '')
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend(loc=legend_loc)
    if caption is not None:
        fig.text(0.5, 0.02, caption, ha='center', va='center', fontsize=11)
    fig.tight_layout()
    fig._qep_noise_robustness = diag
    return fig, ax


def _plot_noise_robustness_combined_right(
    diag_list,
    figsize=(7.4, 5.0),
    title='Noise robustness comparison',
    xlabel_noise='Noise level',
    right_ylabel='Peak-weighted relative error to zero-noise output',
    legend_loc='upper left',
    caption=None,
    right_xscale='log',
    right_xticks=None,
    right_xticklabels=None,
    curve_labels=None,
    curve_colors=None,
    curve_linestyles=None,
    show_boxes=True,
    box_alpha=0.16,
    show_zero_reference=True,
    baseline_color='black',
):
    setup_prab_style()
    if len(diag_list) == 0:
        raise ValueError('diag_list must contain at least one diagnostics dict.')
    if curve_labels is None:
        curve_labels = [diag['method_label'] for diag in diag_list]
    if curve_colors is None:
        curve_colors = [COLORS[diag['method_label']] for diag in diag_list]
    if curve_linestyles is None:
        curve_linestyles = ['-', '--', '-.', ':']
    if not (len(curve_labels) == len(diag_list) == len(curve_colors)):
        raise ValueError('curve_labels/curve_colors must match diag_list length.')

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i, diag in enumerate(diag_list):
        positions = np.asarray(diag['right_noise_levels'], dtype=float)
        all_errors = np.asarray(diag['all_errors_displayed'], dtype=float)
        medians = np.asarray(diag['median_displayed'], dtype=float)
        color = curve_colors[i]
        linestyle = curve_linestyles[i % len(curve_linestyles)]

        if show_boxes:
            if len(positions) == 1:
                widths = [0.18 * positions[0]] if right_xscale == 'log' else [max(0.18 * positions[0], 1e-6)]
            elif right_xscale == 'log':
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
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(box_alpha)

        ax.plot(
            positions, medians,
            color=color, linewidth=2.3, linestyle=linestyle,
            marker='o', markersize=4, label=curve_labels[i],
        )

    ax.set_xscale(right_xscale)
    all_x = np.concatenate([np.asarray(diag['right_noise_levels'], dtype=float) for diag in diag_list])
    if right_xscale == 'log':
        ax.set_xlim(all_x.min() * 0.85, all_x.max() * 1.18)
    else:
        uniq = np.unique(np.sort(all_x))
        if len(uniq) == 1:
            pad = max(0.35 * uniq[0], 1e-6)
        else:
            pad = 0.55 * np.min(np.diff(uniq))
        ax.set_xlim(max(0.0, all_x.min() - pad), all_x.max() + pad)

    if show_zero_reference:
        ax.axhline(0.0, color=baseline_color, linewidth=1.1, linestyle=':')

    if right_xticks is not None:
        ax.set_xticks(right_xticks)
    if right_xticklabels is not None:
        ax.set_xticklabels(right_xticklabels)

    ax.set_xlabel(xlabel_noise)
    ax.set_ylabel(right_ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.25)
    ax.legend(loc=legend_loc)
    if caption is not None:
        fig.text(0.5, 0.02, caption, ha='center', va='center', fontsize=11)
    fig.tight_layout()
    return fig, ax


def _plot_noise_robustness_core(
    method,
    idx,
    val_dataset,
    device,
    dx_um,
    fwhm_center,
    align_by_fwhm,
    model=None,
    forward_spectrum_fft=None,
    gerchberg_saxton_1d_torch=None,
    rep_levels=(1e-4, 1e-3, 1e-2),
    noise_levels=None,
    right_noise_levels=None,
    n_trials=50,
    figsize=(13, 5.2),
    title_left=None,
    title_right=None,
    xlabel_profile='z (μm)',
    xlabel_noise='Noise level',
    left_ylabel='Normalized charge',
    right_ylabel='Peak-weighted relative error to zero-noise output',
    baseline_label='0 noise',
    rep_labels=None,
    legend_loc='upper right',
    caption=None,
    gs_seed_fixed=0,
    gs_iters=2000,
    use_support=True,
    support_width_um=30.0,
    enforce_monotone_display=False,
    box_facecolor=None,
    median_color='black',
    baseline_color='black',
    rep_colors=None,
    rep_linestyles=None,
    right_xscale='log',
    right_xticks=None,
    right_xticklabels=None,
):
    setup_prab_style()
    diag = _compute_noise_robustness_diagnostics(
        method=method, idx=idx, val_dataset=val_dataset, device=device, dx_um=dx_um,
        fwhm_center=fwhm_center, align_by_fwhm=align_by_fwhm, model=model,
        forward_spectrum_fft=forward_spectrum_fft, gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
        rep_levels=rep_levels, noise_levels=noise_levels, right_noise_levels=right_noise_levels,
        n_trials=n_trials, gs_seed_fixed=gs_seed_fixed, gs_iters=gs_iters,
        use_support=use_support, support_width_um=support_width_um,
        enforce_monotone_display=enforce_monotone_display,
    )
    if box_facecolor is None:
        box_facecolor = COLORS[diag['method_label']]
    rep_profiles = diag['rep_profiles']
    if rep_labels is None:
        rep_labels = [f'{nl*100:.2f}% noise' for nl, _ in rep_profiles]
    if rep_colors is None:
        rep_colors = ['#1f77b4', '#e67e22', '#d62728']
    if rep_linestyles is None:
        rep_linestyles = ['--', '-.', ':']

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(bottom=0.18, wspace=0.28)

    ax_left.plot(diag['z_um'], diag['baseline_profile'], color=baseline_color, linestyle='-', linewidth=2.2, label=baseline_label)
    for i, ((_, prof), label) in enumerate(zip(rep_profiles, rep_labels)):
        ax_left.plot(diag['z_um'], prof, color=rep_colors[i % len(rep_colors)], linestyle=rep_linestyles[i % len(rep_linestyles)], linewidth=2.0, label=label)
    ax_left.set_xlabel(xlabel_profile)
    ax_left.set_ylabel(left_ylabel)
    ax_left.set_title(title_left or '')
    ax_left.grid(alpha=0.25)
    ax_left.legend(loc=legend_loc)

    positions = np.asarray(diag['right_noise_levels'], dtype=float)
    all_errors = np.asarray(diag['all_errors_displayed'], dtype=float)
    medians = np.asarray(diag['median_displayed'], dtype=float)

    if len(positions) == 1:
        widths = [0.25 * positions[0]] if right_xscale == 'log' else [max(0.25 * positions[0], 1e-6)]
    elif right_xscale == 'log':
        widths = 0.22 * positions
    else:
        diffs = np.diff(np.sort(positions))
        widths = [0.45 * float(np.min(diffs))] * len(positions)

    bp = ax_right.boxplot(
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
    for patch in bp['boxes']:
        patch.set_facecolor(box_facecolor)
        patch.set_alpha(0.45)

    ax_right.plot(positions, medians, color=median_color, linewidth=2.2, marker='o', markersize=4, label='Median trend')
    ax_right.axhline(0.0, color=baseline_color, linewidth=1.2, linestyle=':', label='Zero-noise reference')
    ax_right.set_xscale(right_xscale)
    if right_xscale == 'log':
        ax_right.set_xlim(positions.min() * 0.85, positions.max() * 1.18)
    else:
        if len(positions) == 1:
            pad = max(0.35 * positions[0], 1e-6)
        else:
            pad = 0.55 * np.min(np.diff(np.sort(positions)))
        ax_right.set_xlim(max(0.0, positions.min() - pad), positions.max() + pad)
    if right_xticks is not None:
        ax_right.set_xticks(right_xticks)
    if right_xticklabels is not None:
        ax_right.set_xticklabels(right_xticklabels)
    ax_right.set_xlabel(xlabel_noise)
    ax_right.set_ylabel(right_ylabel)
    ax_right.set_title(title_right or '')
    ax_right.grid(axis='y', linestyle='--', alpha=0.25)
    ax_right.legend(loc=legend_loc)

    if caption is not None:
        fig.text(0.5, 0.04, caption, ha='center', va='center', fontsize=11)

    fig._qep_noise_robustness = diag
    return fig, (ax_left, ax_right)


def get_noise_robustness_nn_diagnostics(
    model,
    val_dataset,
    idx,
    device,
    dx_um,
    fwhm_center,
    align_by_fwhm,
    rep_levels=(1e-4, 1e-3, 1e-2),
    noise_levels=None,
    right_noise_levels=None,
    n_trials=50,
    enforce_monotone_display=False,
):
    return _compute_noise_robustness_diagnostics(
        method='nn', idx=idx, val_dataset=val_dataset, device=device, dx_um=dx_um,
        fwhm_center=fwhm_center, align_by_fwhm=align_by_fwhm, model=model,
        rep_levels=rep_levels, noise_levels=noise_levels, right_noise_levels=right_noise_levels,
        n_trials=n_trials, enforce_monotone_display=enforce_monotone_display,
    )


def get_noise_robustness_gs_diagnostics(
    val_dataset,
    idx,
    device,
    dx_um,
    fwhm_center,
    align_by_fwhm,
    forward_spectrum_fft,
    gerchberg_saxton_1d_torch,
    rep_levels=(1e-4, 1e-3, 1e-2),
    noise_levels=None,
    right_noise_levels=None,
    n_trials=50,
    gs_seed_fixed=0,
    gs_iters=2000,
    use_support=True,
    support_width_um=30.0,
    enforce_monotone_display=False,
):
    return _compute_noise_robustness_diagnostics(
        method='gs', idx=idx, val_dataset=val_dataset, device=device, dx_um=dx_um,
        fwhm_center=fwhm_center, align_by_fwhm=align_by_fwhm,
        forward_spectrum_fft=forward_spectrum_fft, gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
        rep_levels=rep_levels, noise_levels=noise_levels, right_noise_levels=right_noise_levels,
        n_trials=n_trials, gs_seed_fixed=gs_seed_fixed, gs_iters=gs_iters,
        use_support=use_support, support_width_um=support_width_um,
        enforce_monotone_display=enforce_monotone_display,
    )


def plot_noise_robustness_left_from_diagnostics(
    diag,
    figsize=(6.4, 5.0),
    title=None,
    xlabel_profile='z (μm)',
    left_ylabel='Normalized charge',
    baseline_label='0 noise',
    rep_labels=None,
    legend_loc='upper right',
    baseline_color='black',
    rep_colors=None,
    rep_linestyles=None,
    caption=None,
):
    return _plot_noise_robustness_left_from_diag(
        diag=diag, figsize=figsize, title=title, xlabel_profile=xlabel_profile,
        left_ylabel=left_ylabel, baseline_label=baseline_label, rep_labels=rep_labels,
        legend_loc=legend_loc, baseline_color=baseline_color, rep_colors=rep_colors,
        rep_linestyles=rep_linestyles, caption=caption,
    )


def plot_noise_robustness_right_from_diagnostics(
    diag,
    figsize=(6.8, 5.0),
    title=None,
    xlabel_noise='Noise level',
    right_ylabel='Peak-weighted relative error to zero-noise output',
    legend_loc='upper right',
    caption=None,
    box_facecolor=None,
    median_color='black',
    baseline_color='black',
    right_xscale='log',
    right_xticks=None,
    right_xticklabels=None,
    curve_label='Median trend',
    zero_ref_label='Zero-noise reference',
):
    return _plot_noise_robustness_right_from_diag(
        diag=diag, figsize=figsize, title=title, xlabel_noise=xlabel_noise,
        right_ylabel=right_ylabel, legend_loc=legend_loc, caption=caption,
        box_facecolor=box_facecolor, median_color=median_color, baseline_color=baseline_color,
        right_xscale=right_xscale, right_xticks=right_xticks, right_xticklabels=right_xticklabels,
        curve_label=curve_label, zero_ref_label=zero_ref_label,
    )


def plot_noise_robustness_comparison(
    diag_list,
    figsize=(7.4, 5.0),
    title='Noise robustness comparison',
    xlabel_noise='Noise level',
    right_ylabel='Peak-weighted relative error to zero-noise output',
    legend_loc='upper left',
    caption=None,
    right_xscale='log',
    right_xticks=None,
    right_xticklabels=None,
    curve_labels=None,
    curve_colors=None,
    curve_linestyles=None,
    show_boxes=True,
    box_alpha=0.16,
    show_zero_reference=True,
    baseline_color='black',
):
    return _plot_noise_robustness_combined_right(
        diag_list=diag_list, figsize=figsize, title=title, xlabel_noise=xlabel_noise,
        right_ylabel=right_ylabel, legend_loc=legend_loc, caption=caption,
        right_xscale=right_xscale, right_xticks=right_xticks, right_xticklabels=right_xticklabels,
        curve_labels=curve_labels, curve_colors=curve_colors, curve_linestyles=curve_linestyles,
        show_boxes=show_boxes, box_alpha=box_alpha, show_zero_reference=show_zero_reference,
        baseline_color=baseline_color,
    )


def plot_noise_robustness_nn(
    model,
    val_dataset,
    idx,
    device,
    dx_um,
    fwhm_center,
    align_by_fwhm,
    rep_levels=(1e-4, 1e-3, 1e-2),
    noise_levels=None,
    right_noise_levels=None,
    n_trials=50,
    figsize=(13, 5.2),
    title_left='1D dilated ResNet reconstruction under spectral noise',
    title_right='1D dilated ResNet: deviation from zero-noise reconstruction',
    xlabel_profile='z (μm)',
    xlabel_noise='Noise level',
    left_ylabel='Normalized charge',
    right_ylabel='Peak-weighted relative error to zero-noise output',
    baseline_label='0 noise',
    rep_labels=None,
    legend_loc='upper right',
    caption=None,
    enforce_monotone_display=False,
    baseline_color='black',
    rep_colors=None,
    rep_linestyles=None,
    right_xscale='log',
    right_xticks=None,
    right_xticklabels=None,
):
    return _plot_noise_robustness_core(
        method='nn', idx=idx, val_dataset=val_dataset, device=device, dx_um=dx_um,
        fwhm_center=fwhm_center, align_by_fwhm=align_by_fwhm, model=model,
        rep_levels=rep_levels, noise_levels=noise_levels, right_noise_levels=right_noise_levels,
        n_trials=n_trials, figsize=figsize, title_left=title_left, title_right=title_right,
        xlabel_profile=xlabel_profile, xlabel_noise=xlabel_noise, left_ylabel=left_ylabel,
        right_ylabel=right_ylabel, baseline_label=baseline_label, rep_labels=rep_labels,
        legend_loc=legend_loc, caption=caption, enforce_monotone_display=enforce_monotone_display,
        box_facecolor=COLORS[METHOD_NN], baseline_color=baseline_color,
        rep_colors=rep_colors, rep_linestyles=rep_linestyles, right_xscale=right_xscale,
        right_xticks=right_xticks, right_xticklabels=right_xticklabels,
    )


def plot_noise_robustness_gs(
    val_dataset,
    idx,
    device,
    dx_um,
    fwhm_center,
    align_by_fwhm,
    forward_spectrum_fft,
    gerchberg_saxton_1d_torch,
    rep_levels=(1e-4, 1e-3, 1e-2),
    noise_levels=None,
    right_noise_levels=None,
    n_trials=50,
    figsize=(13, 5.2),
    title_left='GS reconstruction under spectral noise',
    title_right='GS: deviation from zero-noise reconstruction',
    xlabel_profile='z (μm)',
    xlabel_noise='Noise level',
    left_ylabel='Normalized charge',
    right_ylabel='Peak-weighted relative error to zero-noise output',
    baseline_label='0 noise',
    rep_labels=None,
    legend_loc='upper right',
    caption=None,
    gs_seed_fixed=0,
    gs_iters=2000,
    use_support=True,
    support_width_um=30.0,
    enforce_monotone_display=False,
    baseline_color='black',
    rep_colors=None,
    rep_linestyles=None,
    right_xscale='log',
    right_xticks=None,
    right_xticklabels=None,
):
    return _plot_noise_robustness_core(
        method='gs', idx=idx, val_dataset=val_dataset, device=device, dx_um=dx_um,
        fwhm_center=fwhm_center, align_by_fwhm=align_by_fwhm,
        forward_spectrum_fft=forward_spectrum_fft, gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
        rep_levels=rep_levels, noise_levels=noise_levels, right_noise_levels=right_noise_levels,
        n_trials=n_trials, figsize=figsize, title_left=title_left, title_right=title_right,
        xlabel_profile=xlabel_profile, xlabel_noise=xlabel_noise, left_ylabel=left_ylabel,
        right_ylabel=right_ylabel, baseline_label=baseline_label, rep_labels=rep_labels,
        legend_loc=legend_loc, caption=caption, gs_seed_fixed=gs_seed_fixed, gs_iters=gs_iters,
        use_support=use_support, support_width_um=support_width_um,
        enforce_monotone_display=enforce_monotone_display, box_facecolor=COLORS[METHOD_GS],
        baseline_color=baseline_color, rep_colors=rep_colors, rep_linestyles=rep_linestyles,
        right_xscale=right_xscale, right_xticks=right_xticks, right_xticklabels=right_xticklabels,
    )

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
