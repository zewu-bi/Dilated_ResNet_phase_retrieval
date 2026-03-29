
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from src.reconstruction_utils import align_by_fwhm_nonperiodic

METHOD_NN = "1D dilated resnet"
METHOD_GS = "GS"
GROUP_ORDER = ["Single", "Double"]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def finite_array(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def safe_global_idx(val_dataset, local_idx):
    if hasattr(val_dataset, "indices"):
        return int(val_dataset.indices[local_idx])
    return int(local_idx)


def centered_axis(n, dx_um):
    return (np.arange(int(n)) - int(n) // 2) * float(dx_um)


def positive_axis(n, dx_um):
    return np.arange(int(n)) * float(dx_um)


def l1_nonneg_normalize(y):
    y = np.clip(to_numpy(y), 0, None).astype(np.float64)
    s = y.sum()
    return y / s if s > 0 else np.zeros_like(y)


def max_normalize(y):
    y = np.clip(to_numpy(y), 0, None).astype(np.float64)
    m = y.max()
    return y / m if m > 0 else y


def safe_fwhm_center(y, z, fwhm_center):
    y = max_normalize(y)
    try:
        return float(fwhm_center(y, z))
    except Exception:
        s = y.sum()
        return float((z * y).sum() / s) if s > 0 else float(z[len(z) // 2])


def align_profile_to_target_center(pred, true, z, dx_um, fwhm_center, align_by_fwhm, nonperiodic=False):
    c_true = safe_fwhm_center(true, z, fwhm_center)
    pred_np = to_numpy(pred)
    if nonperiodic:
        return to_numpy(align_by_fwhm_nonperiodic(pred_np, z, c_true, dx_um))
    try:
        return to_numpy(align_by_fwhm(pred_np, z, c_true, dx_um))
    except Exception:
        return pred_np


def profile_nrmse(pred, true):
    pred = to_numpy(pred)
    true = to_numpy(true)
    denom = np.linalg.norm(true)
    return np.linalg.norm(pred - true) / denom if denom > 0 else np.nan


def compute_band_spectral_error(profile, measured_band, dx_um, f_min, f_max, pad_factor,
                                compute_form_factor, k_to_THz):
    profile = l1_nonneg_normalize(profile)
    measured_band = np.clip(to_numpy(measured_band), 0, None).astype(np.float64)

    z = positive_axis(len(profile), dx_um)
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


def find_two_peak_indices(y, dx_um, min_sep_um=2.0, min_height_rel=0.15):
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
    chosen = find_two_peak_indices(y, dx_um, min_sep_um=min_sep_um, min_height_rel=min_height_rel)
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
    z_um = centered_axis(N, dx_um)
    mask_np = (np.abs(z_um) <= width_um / 2).astype(np.float32)
    return torch.tensor(mask_np, dtype=torch.float32, device=device)


def maybe_pair_from_meta(sample_meta, candidate_keys):
    if not isinstance(sample_meta, dict):
        return None
    for key in candidate_keys:
        if key in sample_meta:
            val = sample_meta[key]
            arr = np.asarray(val, dtype=float).reshape(-1)
            if arr.size >= 2:
                return arr[:2]
    return None


def weighted_sigma(z, y):
    y = np.clip(np.asarray(y, dtype=float), 0, None)
    s = y.sum()
    if s <= 0:
        return np.nan
    mu = float((z * y).sum() / s)
    var = float(((z - mu) ** 2 * y).sum() / s)
    return np.sqrt(max(var, 0.0))


def estimate_double_parameters(sample_meta, target, z_centered, dx_um):
    charges = maybe_pair_from_meta(
        sample_meta,
        ["charges", "charge_list", "q_list", "qs", "amplitudes", "weights"],
    )
    centers = maybe_pair_from_meta(
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
    peaks = find_two_peak_indices(y, dx_um=dx_um, min_sep_um=2.0, min_height_rel=0.10)

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
    centers = maybe_pair_from_meta(
        sample_meta,
        ["centers", "means", "mus", "mu_list", "positions", "pos", "x0", "centroids"],
    )
    sigmas = maybe_pair_from_meta(
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
    peaks = find_two_peak_indices(y, dx_um=dx_um, min_sep_um=2.0, min_height_rel=0.10)
    if len(peaks) < 2:
        return np.nan, np.nan

    valley_rel = np.argmin(y[peaks[0]: peaks[1] + 1])
    valley = int(peaks[0] + valley_rel)

    left_mask = np.zeros_like(y, dtype=bool)
    right_mask = np.zeros_like(y, dtype=bool)
    left_mask[: valley + 1] = True
    right_mask[valley + 1:] = True

    sigma_left = weighted_sigma(z_centered[left_mask], y[left_mask])
    sigma_right = weighted_sigma(z_centered[right_mask], y[right_mask])
    return float(sigma_left), float(sigma_right)


def select_random_local_indices(
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
        global_idx = safe_global_idx(val_dataset, local_idx)
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


def run_gs_once(
    I_meas,
    gerchberg_saxton_1d_torch,
    seed,
    device,
    n_iters,
    support_mask,
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
    return l1_nonneg_normalize(rho_gs.detach().cpu().numpy()), (t1 - t0) * 1000.0


def postprocess_gs_prediction(
    pred_gs,
    target,
    z,
    dx_um,
    fwhm_center,
    align_by_fwhm,
    group,
    allow_gs_double_swap=False,
):
    target_np = l1_nonneg_normalize(target)
    pred_np = l1_nonneg_normalize(pred_gs)

    if not (allow_gs_double_swap and group == "Double"):
        aligned = align_profile_to_target_center(
            pred_np, target_np, z, dx_um, fwhm_center, align_by_fwhm, nonperiodic=False
        )
        aligned = l1_nonneg_normalize(aligned)
        return aligned, {
            "was_flipped": False,
            "alignment_mode": "legacy_circular",
            "candidate_error": profile_nrmse(aligned, target_np),
        }

    aligned_direct = align_profile_to_target_center(
        pred_np, target_np, z, dx_um, fwhm_center, align_by_fwhm, nonperiodic=True
    )
    aligned_direct = l1_nonneg_normalize(aligned_direct)

    flipped = pred_np[::-1].copy()
    aligned_flipped = align_profile_to_target_center(
        flipped, target_np, z, dx_um, fwhm_center, align_by_fwhm, nonperiodic=True
    )
    aligned_flipped = l1_nonneg_normalize(aligned_flipped)

    err_direct = profile_nrmse(aligned_direct, target_np)
    err_flipped = profile_nrmse(aligned_flipped, target_np)

    if np.isfinite(err_flipped) and (not np.isfinite(err_direct) or err_flipped < err_direct):
        return aligned_flipped, {
            "was_flipped": True,
            "alignment_mode": "nonperiodic_flip_test",
            "candidate_error": err_flipped,
        }

    return aligned_direct, {
        "was_flipped": False,
        "alignment_mode": "nonperiodic_flip_test",
        "candidate_error": err_direct,
    }


def set_torch_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def normalize_profile(y):
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 0.0, None)
    s = y.sum()
    if s <= 0:
        return np.zeros_like(y)
    return y / s


def weighted_relative_error(pred, ref, weight):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    weight = np.asarray(weight, dtype=float)
    num = np.sqrt(np.sum(weight * (pred - ref) ** 2))
    den = np.sqrt(np.sum(weight * ref ** 2))
    return num / den if den > 0 else np.nan


def maybe_align_profile(pred, z, c_ref, dx_um, align_by_fwhm):
    try:
        return np.asarray(align_by_fwhm(pred, z, c_ref, dx_um), dtype=float)
    except Exception:
        return np.asarray(pred, dtype=float)


def reconstruct_nn_from_band(model, img_band):
    with torch.no_grad():
        pred = model(img_band).squeeze().detach().cpu().numpy()
    return normalize_profile(pred)


def reconstruct_gs_from_profile(
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

    set_torch_seed(noise_seed)
    noise = nl * torch.randn_like(I_meas)
    I_noisy = torch.clamp(I_meas + noise, min=0.0)
    I_noisy = I_noisy / torch.clamp(torch.max(I_noisy), min=1e-12)

    support_mask = None
    if use_support:
        support_mask = make_support_mask(len(tgt_profile), dx_um, support_width_um, device)

    set_torch_seed(gs_seed)
    rho_gs = gerchberg_saxton_1d_torch(
        I_meas=I_noisy,
        n_iters=gs_iters,
        support_mask=support_mask,
        smooth=True,
        device=device,
    )
    return normalize_profile(rho_gs.detach().cpu().numpy())


def force_monotone_by_trial(error_curves_2d):
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
        raise ValueError("right_noise_levels must be strictly positive.")

    img, tgt = val_dataset[int(idx)]
    img = img.unsqueeze(0).to(device)
    tgt = tgt.squeeze().detach().cpu().numpy()
    z = positive_axis(len(tgt), dx_um)
    tgt_norm = normalize_profile(tgt)

    if method == "nn":
        if model is None:
            raise ValueError('model must be provided for method="nn".')
        model.eval()
        base_profile = reconstruct_nn_from_band(model, img)
        method_label = METHOD_NN
    elif method == "gs":
        if forward_spectrum_fft is None or gerchberg_saxton_1d_torch is None:
            raise ValueError('forward_spectrum_fft and gerchberg_saxton_1d_torch must be provided for method="gs".')
        base_profile = reconstruct_gs_from_profile(
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
    base_aligned = maybe_align_profile(base_profile, z, c_ref, dx_um, align_by_fwhm)
    weight = base_aligned / max(np.max(base_aligned), 1e-12)

    rep_profiles = []
    for nl in rep_levels:
        if method == "nn":
            set_torch_seed(0)
            noise = nl * torch.randn_like(img)
            img_noisy = torch.clamp(img + noise, min=0.0)
            pred = reconstruct_nn_from_band(model, img_noisy)
        else:
            pred = reconstruct_gs_from_profile(
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
        pred = maybe_align_profile(pred, z, c_ref, dx_um, align_by_fwhm)
        rep_profiles.append((float(nl), pred))

    all_errors = []
    for nl in right_noise_levels:
        trial_errors = []
        for seed in range(int(n_trials)):
            if method == "nn":
                set_torch_seed(seed)
                noise = nl * torch.randn_like(img)
                img_noisy = torch.clamp(img + noise, min=0.0)
                pred = reconstruct_nn_from_band(model, img_noisy)
                pred = maybe_align_profile(pred, z, c_ref, dx_um, align_by_fwhm)
            else:
                pred = reconstruct_gs_from_profile(
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
                pred = maybe_align_profile(pred, z, c_ref, dx_um, align_by_fwhm)
            trial_errors.append(weighted_relative_error(pred, base_aligned, weight))
        all_errors.append(trial_errors)

    all_errors = np.asarray(all_errors, dtype=float)
    raw_errors = all_errors.copy()
    if enforce_monotone_display:
        all_errors = force_monotone_by_trial(all_errors)
    medians = np.median(all_errors, axis=1)

    return {
        "idx": int(idx),
        "method": method,
        "method_label": method_label,
        "z_um": z,
        "target": tgt_norm,
        "baseline_profile": base_aligned,
        "rep_profiles": rep_profiles,
        "right_noise_levels": right_noise_levels.astype(float),
        "all_errors_displayed": all_errors,
        "all_errors_raw": raw_errors,
        "median_displayed": medians,
        "n_trials": int(n_trials),
        "enforce_monotone_display": bool(enforce_monotone_display),
        "gs_seed_fixed": int(gs_seed_fixed),
        "gs_iters": int(gs_iters),
        "use_support": bool(use_support),
        "support_width_um": float(support_width_um),
    }


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
        method="nn",
        idx=idx,
        val_dataset=val_dataset,
        device=device,
        dx_um=dx_um,
        fwhm_center=fwhm_center,
        align_by_fwhm=align_by_fwhm,
        model=model,
        rep_levels=rep_levels,
        noise_levels=noise_levels,
        right_noise_levels=right_noise_levels,
        n_trials=n_trials,
        enforce_monotone_display=enforce_monotone_display,
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
        method="gs",
        idx=idx,
        val_dataset=val_dataset,
        device=device,
        dx_um=dx_um,
        fwhm_center=fwhm_center,
        align_by_fwhm=align_by_fwhm,
        forward_spectrum_fft=forward_spectrum_fft,
        gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
        rep_levels=rep_levels,
        noise_levels=noise_levels,
        right_noise_levels=right_noise_levels,
        n_trials=n_trials,
        gs_seed_fixed=gs_seed_fixed,
        gs_iters=gs_iters,
        use_support=use_support,
        support_width_um=support_width_um,
        enforce_monotone_display=enforce_monotone_display,
    )


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
    allow_gs_double_swap=False,
):
    model.eval()

    selection = select_random_local_indices(
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

    if max_per_group is not None:
        limited = []
        counts = defaultdict(int)
        for idx in selected_local_indices:
            group = group_by_local[idx]
            if counts[group] < int(max_per_group):
                limited.append(idx)
                counts[group] += 1
        selected_local_indices = limited

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
        z = positive_axis(N, dx_um)
        z_centered = centered_axis(N, dx_um)

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
        gs_flip_flags = []

        for seed in gs_seeds:
            rho_gs_raw, gs_ms = run_gs_once(
                I_meas=I_meas,
                gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
                seed=seed,
                device=device,
                n_iters=gs_iters,
                support_mask=support_mask,
            )
            rho_gs, flip_meta = postprocess_gs_prediction(
                pred_gs=rho_gs_raw,
                target=tgt_np,
                z=z,
                dx_um=dx_um,
                fwhm_center=fwhm_center,
                align_by_fwhm=align_by_fwhm,
                group=group,
                allow_gs_double_swap=allow_gs_double_swap,
            )
            gs_preds.append(rho_gs)
            gs_time_this_sample.append(gs_ms)
            gs_flip_flags.append(bool(flip_meta["was_flipped"]))

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

        valid_gs_profile_errs = finite_array(gs_profile_errs)
        valid_gs_spec_errs = finite_array(gs_spec_errs)
        valid_gs_struct_errs = finite_array(gs_struct_errs)

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
        dense_flip_fraction = np.nan
        if local_idx in vis_set:
            dense_runs = []
            dense_flip_flags = []
            for seed in dense_vis_seeds:
                rho_gs_raw, _ = run_gs_once(
                    I_meas=I_meas,
                    gerchberg_saxton_1d_torch=gerchberg_saxton_1d_torch,
                    seed=seed,
                    device=device,
                    n_iters=gs_iters,
                    support_mask=support_mask,
                )
                rho_gs, flip_meta = postprocess_gs_prediction(
                    pred_gs=rho_gs_raw,
                    target=tgt_np,
                    z=z,
                    dx_um=dx_um,
                    fwhm_center=fwhm_center,
                    align_by_fwhm=align_by_fwhm,
                    group=group,
                    allow_gs_double_swap=allow_gs_double_swap,
                )
                dense_runs.append(rho_gs)
                dense_flip_flags.append(bool(flip_meta["was_flipped"]))
            if len(dense_runs) > 0:
                dense_gs_preds = np.vstack(dense_runs)
                dense_q = {
                    "q05": np.percentile(dense_gs_preds, 5, axis=0),
                    "q25": np.percentile(dense_gs_preds, 25, axis=0),
                    "q50": np.percentile(dense_gs_preds, 50, axis=0),
                    "q75": np.percentile(dense_gs_preds, 75, axis=0),
                    "q95": np.percentile(dense_gs_preds, 95, axis=0),
                }
                dense_flip_fraction = float(np.mean(dense_flip_flags))

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
            "gs_flip_flags": list(gs_flip_flags),
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
            "dense_gs_flip_fraction": dense_flip_fraction,
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
                vals = finite_array(metric_dict[(group, method)])
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
        vals = finite_array(timing[method])
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
            "allow_gs_double_swap": bool(allow_gs_double_swap),
            "double_heatmap_definition": {
                "x": "Left-bunch effective sigma (μm)",
                "y": "Right-bunch effective sigma (μm)",
                "score": "GS median profile NRMSE - NN profile NRMSE; positive means NN is better",
            },
        },
    }


def build_interpolated_advantage_map(
    cache,
    grid_size=(220, 220),
    method="linear",
    fill_nearest=True,
    smooth_sigma=0.0,
    x_range=None,
    y_range=None,
):
    records = cache.get("double_advantage_records", [])

    xs = np.array([r["sigma_left_um"] for r in records], dtype=float)
    ys = np.array([r["sigma_right_um"] for r in records], dtype=float)
    scores = np.array([r["advantage"] for r in records], dtype=float)

    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(scores)
    xs = xs[finite]
    ys = ys[finite]
    scores = scores[finite]

    if len(xs) == 0:
        return {
            "x": np.array([]),
            "y": np.array([]),
            "z": np.array([[]]),
            "points_x": xs,
            "points_y": ys,
            "points_score": scores,
        }

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

    nx, ny = int(grid_size[0]), int(grid_size[1])
    grid_x = np.linspace(x_range[0], x_range[1], nx)
    grid_y = np.linspace(y_range[0], y_range[1], ny)
    gx, gy = np.meshgrid(grid_x, grid_y)

    points = np.column_stack([xs, ys])
    grid_z = griddata(points, scores, (gx, gy), method=method)

    if fill_nearest and np.any(~np.isfinite(grid_z)):
        grid_nearest = griddata(points, scores, (gx, gy), method="nearest")
        grid_z = np.where(np.isfinite(grid_z), grid_z, grid_nearest)

    if smooth_sigma is not None and float(smooth_sigma) > 0:
        grid_z = gaussian_filter(grid_z, sigma=float(smooth_sigma), mode="nearest")

    return {
        "x": grid_x,
        "y": grid_y,
        "z": grid_z,
        "points_x": xs,
        "points_y": ys,
        "points_score": scores,
        "x_range": x_range,
        "y_range": y_range,
        "method": method,
        "fill_nearest": bool(fill_nearest),
        "smooth_sigma": float(smooth_sigma),
    }


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
    print(f"\nallow_gs_double_swap = {cache['config'].get('allow_gs_double_swap', False)}")
