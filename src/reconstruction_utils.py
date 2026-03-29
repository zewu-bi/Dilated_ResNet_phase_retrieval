
import numpy as np
import torch
import torch.nn.functional as F
import torch.fft as fft
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


def fwhm_center(profile, x):
    if np.max(profile) <= 0:
        return np.mean(x)

    p = gaussian_filter1d(profile, sigma=1)
    peak_idx = np.argmax(p)
    peak_val = p[peak_idx]
    half_val = 0.5 * peak_val

    left_idx = peak_idx
    while left_idx > 0 and p[left_idx] > half_val:
        left_idx -= 1

    right_idx = peak_idx
    while right_idx < len(p) - 1 and p[right_idx] > half_val:
        right_idx += 1

    return 0.5 * (x[left_idx] + x[right_idx])


def align_by_fwhm(profile, x, reference_center, dx_um):
    profile_max = profile / np.max(profile) if np.max(profile) > 0 else profile
    c = fwhm_center(profile_max, x)
    shift_pts = int(np.round((c - reference_center) / dx_um))
    return np.roll(profile, -shift_pts)


def shift_with_zero_fill(profile, shift_pts):
    """
    Shift a 1D profile with zero padding instead of periodic wrapping.

    Positive shift_pts moves the profile to the right.
    Negative shift_pts moves the profile to the left.
    """
    arr = np.asarray(profile)
    shift_pts = int(shift_pts)
    out = np.zeros_like(arr)

    if shift_pts == 0:
        return arr.copy()

    n = arr.shape[0]
    if shift_pts >= n or shift_pts <= -n:
        return out

    if shift_pts > 0:
        out[shift_pts:] = arr[: n - shift_pts]
    else:
        out[: n + shift_pts] = arr[-shift_pts:]

    return out


def align_by_fwhm_nonperiodic(profile, x, reference_center, dx_um):
    """
    Same intent as align_by_fwhm, but uses zero-filled shifting so the two
    ends of the axis are not treated as connected.
    """
    arr = np.asarray(profile)
    profile_max = arr / np.max(arr) if np.max(arr) > 0 else arr
    c = fwhm_center(profile_max, x)
    shift_pts = int(np.round((c - reference_center) / dx_um))
    return shift_with_zero_fill(arr, -shift_pts)


def weighted_error(pred, target, weight):
    diff = weight * (pred - target)
    num = np.linalg.norm(diff)
    den = np.linalg.norm(weight * target)
    return num / den


def _smooth_1d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    x = x.view(1, 1, -1)
    k = kernel.view(1, 1, -1)
    pad = (k.shape[-1] - 1) // 2
    x_pad = F.pad(x, (pad, pad), mode="reflect")
    y = F.conv1d(x_pad, k)
    return y.view(-1)


def get_smoothing_kernel(device="cpu"):
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device)
    k = k / k.sum()
    return k


def gerchberg_saxton_1d_torch(
    I_meas: torch.Tensor,
    n_iters: int = 500,
    support_mask: torch.Tensor = None,
    smooth: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    device = torch.device(device)
    I_meas = I_meas.to(device)
    N = I_meas.numel()

    mag_meas = torch.sqrt(torch.clamp(I_meas, min=0.0))

    rho = torch.rand(N, device=device)
    rho = rho / rho.sum()

    kernel = get_smoothing_kernel(device) if smooth else None

    for _ in range(n_iters):
        Fk = fft.fft(rho)
        phase = torch.exp(1j * torch.angle(Fk))
        F_new = mag_meas * phase

        rho_new = fft.ifft(F_new).real
        rho_new = torch.clamp(rho_new, min=0.0)

        if support_mask is not None:
            rho_new = rho_new * support_mask.to(device)

        if kernel is not None:
            rho_new = _smooth_1d(rho_new, kernel)

        s = rho_new.sum()
        if s > 0:
            rho_new = rho_new / s

        rho = rho_new

    return rho


def get_smoothing_kernel_numpy():
    k = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    k = k / k.sum()
    return k


def _smooth_1d_numpy(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = (kernel.size - 1) // 2
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    y = np.convolve(x_pad, kernel, mode="valid")
    return y


def gerchberg_saxton_1d_numpy(
    I_meas_np: np.ndarray,
    n_iters: int = 500,
    support_mask_np: np.ndarray | None = None,
    smooth: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    I_meas_np = np.asarray(I_meas_np, dtype=np.float64)
    N = I_meas_np.size
    mag_meas = np.sqrt(np.clip(I_meas_np, 0.0, None))

    if support_mask_np is not None:
        support_mask_np = np.asarray(support_mask_np, dtype=np.float64)

    rng = np.random.default_rng(seed)
    rho = rng.random(N)
    rho /= rho.sum()

    kernel = get_smoothing_kernel_numpy() if smooth else None

    for _ in range(n_iters):
        Fk = np.fft.fft(rho)
        phase = np.exp(1j * np.angle(Fk))
        F_new = mag_meas * phase

        rho_new = np.fft.ifft(F_new).real
        rho_new = np.maximum(rho_new, 0.0)

        if support_mask_np is not None:
            rho_new *= support_mask_np

        if kernel is not None and kernel.size > 1:
            rho_new = _smooth_1d_numpy(rho_new, kernel)

        s = rho_new.sum()
        if s > 0:
            rho_new /= s

        rho = rho_new

    return rho


def forward_spectrum_fft(rho: torch.Tensor):
    Fk = torch.fft.fft(rho)
    I = torch.abs(Fk) ** 2
    return I, Fk


def gerchberg_saxton_multistart(
    I_meas: torch.Tensor,
    n_iters: int,
    support_mask: torch.Tensor = None,
    n_restarts: int = 8,
    backend: str = "torch",
    device: str = "cpu",
    seed=None,
) -> torch.Tensor:
    backend = backend.lower()
    device = torch.device(device)

    if seed is not None:
        torch.manual_seed(seed)

    if backend == "torch":
        best_rho = None
        best_loss = float("inf")
        dev = torch.device(device)

        for _ in range(n_restarts):
            rho = gerchberg_saxton_1d_torch(
                I_meas=I_meas,
                n_iters=n_iters,
                support_mask=support_mask,
                smooth=True,
                device=device,
            )

            I_rec, _ = forward_spectrum_fft(rho)
            loss = torch.mean(
                (torch.log(I_rec + 1e-12) - torch.log(I_meas.to(device) + 1e-12)) ** 2
            )
            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_rho = rho.detach()

        print(f"[GS-Torch-{dev.type}] best spectral loss over {n_restarts} restarts: {best_loss:.3e}")
        return best_rho.to(device)

    elif backend == "numpy":
        I_meas_np = I_meas.detach().cpu().numpy()
        support_np = None if support_mask is None else support_mask.detach().cpu().numpy()

        best_rho_np = None
        best_loss = np.inf

        for r in range(n_restarts):
            rho_np = gerchberg_saxton_1d_numpy(
                I_meas_np=I_meas_np,
                n_iters=n_iters,
                support_mask_np=support_np,
                smooth=True,
                seed=r,
            )

            Fk = np.fft.fft(rho_np)
            I_rec_np = np.abs(Fk) ** 2
            loss = np.mean(
                (np.log(I_rec_np + 1e-12) - np.log(I_meas_np + 1e-12)) ** 2
            )

            if loss < best_loss:
                best_loss = float(loss)
                best_rho_np = rho_np.copy()

        print(f"[GS-NumPy] best spectral loss over {n_restarts} restarts: {best_loss:.3e}")
        best_rho_torch = torch.from_numpy(best_rho_np).to(device, dtype=I_meas.dtype)
        return best_rho_torch

    else:
        raise ValueError("backend must be 'torch' or 'numpy'")


def resample_linear(x, y, n=256):
    f_interp = interp1d(x, y, kind="linear", fill_value="extrapolate")
    x_new = np.linspace(np.min(x), np.max(x), n)
    y_new = f_interp(x_new)
    return x_new, y_new
