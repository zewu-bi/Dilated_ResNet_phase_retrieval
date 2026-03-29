
import os

import numpy as np
import pandas as pd
import torch

from src.quant_eval_utils import l1_nonneg_normalize, positive_axis


def prepare_spectrum_payload(
    profile,
    measured_band,
    dx_um,
    f_min,
    f_max,
    f_display_max,
    pad_factor,
    compute_form_factor,
    k_to_THz,
):
    profile = l1_nonneg_normalize(profile)
    z_um = positive_axis(len(profile), dx_um)

    k_rec, F2_rec = compute_form_factor(z_um, profile, pad_factor=pad_factor)
    freq_rec = k_to_THz(k_rec)

    mask_full = (freq_rec >= 0) & (freq_rec <= f_display_max)
    freq_full = freq_rec[mask_full]
    F2_full = F2_rec[mask_full]

    measured_band = np.asarray(measured_band, dtype=float)
    freq_meas = np.linspace(f_min, f_max, len(measured_band))

    return {
        "z_um": z_um,
        "profile": profile,
        "freq_meas": freq_meas,
        "measured_band": measured_band,
        "freq_full": freq_full,
        "reconstructed_full": F2_full,
        "f_min": float(f_min),
        "f_max": float(f_max),
        "f_display_max": float(f_display_max),
    }


def prepare_validation_case(
    model,
    val_dataset,
    idx,
    device,
    dx_um,
    f_min,
    f_max,
    f_display_max,
    pad_factor,
    compute_form_factor,
    k_to_THz,
):
    img, tgt = val_dataset[int(idx)]
    img_model = img.unsqueeze(0).to(device)

    target = l1_nonneg_normalize(tgt.squeeze().cpu().numpy())

    with torch.no_grad():
        pred = model(img_model).squeeze().cpu().numpy()
    prediction = l1_nonneg_normalize(pred)

    payload = prepare_spectrum_payload(
        profile=prediction,
        measured_band=img.squeeze().cpu().numpy(),
        dx_um=dx_um,
        f_min=f_min,
        f_max=f_max,
        f_display_max=f_display_max,
        pad_factor=pad_factor,
        compute_form_factor=compute_form_factor,
        k_to_THz=k_to_THz,
    )
    payload["target"] = target
    payload["prediction"] = prediction
    payload["idx"] = int(idx)
    return payload


def load_bubblewrap_cases(
    folder,
    dx_um,
    f_min,
    f_max,
    target_points,
    resample_fn,
    file_order=None,
):
    if file_order is None:
        file_order = [
            "F1.csv", "F2.csv", "F3.csv",
            "rho1.csv", "rho2.csv", "rho3.csv",
        ]

    F_list = []
    rho_list = []

    for filename in file_order:
        path = os.path.join(folder, filename)
        data = pd.read_csv(path)

        x = data.iloc[:, 0].values.astype(float)
        y = data.iloc[:, 1].values.astype(float)

        if filename.startswith("F"):
            omega = x * 1e14
            freq_THz = omega / (2 * np.pi) / 1e12

            mask = (freq_THz >= f_min) & (freq_THz <= f_max)
            freq_crop = freq_THz[mask]
            y_crop = y[mask]

            y_crop = y_crop / np.max(np.abs(y_crop))
            _, y_res = resample_fn(freq_crop, y_crop, target_points)
            F_list.append(y_res.copy())
        else:
            index_coord = x / dx_um
            index_coord = np.round(index_coord).astype(int)

            sort_idx = np.argsort(index_coord)
            index_coord = index_coord[sort_idx]
            y_sorted = y[sort_idx]

            idx_min = index_coord.min()
            idx_max = index_coord.max()

            full_index = np.arange(idx_min, idx_max + 1)
            full_rho = np.zeros_like(full_index, dtype=float)
            full_rho[index_coord - idx_min] = y_sorted

            full_rho = full_rho / np.max(np.abs(full_rho))

            current_len = len(full_rho)
            if current_len > target_points:
                start = (current_len - target_points) // 2
                full_rho = full_rho[start:start + target_points]
            else:
                pad_total = target_points - current_len
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                full_rho = np.pad(full_rho, (pad_left, pad_right))

            rho_list.append(full_rho.copy())

    cases = []
    n_cases = min(len(F_list), len(rho_list))
    for i in range(n_cases):
        amplitude = np.asarray(F_list[i], dtype=float)
        power = amplitude * amplitude
        target_profile = l1_nonneg_normalize(rho_list[i])
        cases.append({
            "name": f"Profile {i + 1}",
            "band_amplitude": amplitude,
            "band_power": power,
            "target_profile": target_profile,
        })

    return cases


def prepare_bubblewrap_case(
    model,
    band_power,
    target_profile,
    device,
    dx_um,
    f_min,
    f_max,
    f_display_max,
    pad_factor,
    compute_form_factor,
    k_to_THz,
):
    img_model = torch.tensor(band_power, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_model).squeeze().cpu().numpy()
    prediction = l1_nonneg_normalize(pred)

    payload = prepare_spectrum_payload(
        profile=prediction,
        measured_band=np.asarray(band_power, dtype=float),
        dx_um=dx_um,
        f_min=f_min,
        f_max=f_max,
        f_display_max=f_display_max,
        pad_factor=pad_factor,
        compute_form_factor=compute_form_factor,
        k_to_THz=k_to_THz,
    )
    payload["target"] = l1_nonneg_normalize(target_profile)
    payload["prediction"] = prediction
    return payload
