import numpy as np
import matplotlib.pyplot as plt


def l1_nonneg_normalize(x):
    x = np.clip(x, 0, None)
    s = np.sum(x)
    return x / s if s > 0 else x


def compute_form_factor(x_um, y, pad_factor=10):
    """
    Compute form factor power spectrum |F(k)|^2 with zero-padding
    to increase frequency-domain resolution.

    pad_factor = how many times to extend array length via zero-padding.
    """
    y_norm = y / np.trapezoid(y, x_um)
    N = len(y_norm)
    Np = pad_factor * N
    y_pad = np.zeros(Np)
    y_pad[:N] = y_norm

    Y = np.fft.fft(y_pad)

    dx = x_um[1] - x_um[0]
    k = np.fft.fftfreq(Np, d=dx) * 2 * np.pi

    F2 = np.abs(Y) ** 2
    return k[:Np // 2], F2[:Np // 2]


def k_to_THz(k):
    """
    Convert wavenumber k (1/um) to frequency in THz.
    """
    c = 3e8
    k_m = k * 1e6
    return (c * k_m / (2 * np.pi)) / 1e12


def plot_bunch_and_form_factor(x_um, y, f_cut_THz=100, pad_factor=10):
    """
    Plot bunch profile and |F|^2 with zero-padding to boost resolution.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x_um, y)
    plt.xlabel("x (um)")
    plt.ylabel("Intensity")
    plt.title("Bunch Profile")
    plt.grid()
    plt.show()

    k, F2 = compute_form_factor(x_um, y, pad_factor=pad_factor)
    freq_THz = k_to_THz(k)

    mask = (freq_THz >= 0) & (freq_THz <= f_cut_THz)

    plt.figure(figsize=(8, 4))
    plt.plot(freq_THz[mask], F2[mask])
    plt.xlabel("Frequency (THz)")
    plt.ylabel("|F(f)|²")
    plt.title(f"Form Factor |F|² (Zero-padding ×{pad_factor}) 0–{f_cut_THz} THz")
    plt.grid()
    plt.show()


def resample_to_fixed_length(x, y, out_len=1024):
    x_new = np.linspace(x.min(), x.max(), out_len)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new
