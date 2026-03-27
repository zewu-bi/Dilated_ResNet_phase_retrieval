from __future__ import annotations

import io
import __main__
import os
from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.interpolate import interp1d

# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "Dilated_ResNet_200.pth"))
TARGET_POINTS = 256
BAND_MIN_THZ = 50.0
BAND_MAX_THZ = 230.0
DX_UM = 0.15
DEVICE = torch.device("cpu")


# -----------------------------
# Model definition
# -----------------------------
class DilatedResBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


class Dilated_CNN_ResNet(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 64):
        super().__init__()
        self.in_conv = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)
        self.block1 = DilatedResBlock(base_ch, dilation=1)
        self.block2 = DilatedResBlock(base_ch, dilation=2)
        self.block3 = DilatedResBlock(base_ch, dilation=4)
        self.block4 = DilatedResBlock(base_ch, dilation=8)
        self.block5 = DilatedResBlock(base_ch, dilation=16)
        self.block6 = DilatedResBlock(base_ch, dilation=32)
        self.out_conv = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return self.out_conv(x)


# Your .pth was saved from a notebook __main__ context.
setattr(__main__, "DilatedResBlock", DilatedResBlock)
setattr(__main__, "Dilated_CNN_ResNet", Dilated_CNN_ResNet)


@lru_cache(maxsize=1)
def load_model() -> nn.Module:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    loaded = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(loaded, nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        model = Dilated_CNN_ResNet(in_ch=1, base_ch=64)
        model.load_state_dict(loaded)
    else:
        raise RuntimeError(f"Unsupported model payload type: {type(loaded)}")

    model.to(DEVICE)
    model.eval()
    return model


# -----------------------------
# Preprocessing
# -----------------------------
def _read_csv_bytes(contents: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(contents), header=None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="CSV must contain at least two columns.")

    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="CSV does not contain enough numeric rows.")
    return df


def _infer_frequency_thz(x: np.ndarray) -> np.ndarray:
    # Notebook convention: first column stores omega scaled by 1e14.
    freq_from_omega = x * 1e14 / (2 * np.pi) / 1e12
    in_band_omega = np.count_nonzero((freq_from_omega >= BAND_MIN_THZ) & (freq_from_omega <= BAND_MAX_THZ))
    in_band_direct = np.count_nonzero((x >= BAND_MIN_THZ) & (x <= BAND_MAX_THZ))

    if in_band_omega >= max(8, in_band_direct):
        return freq_from_omega
    return x


def preprocess_spectrum(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)

    freq_thz = _infer_frequency_thz(x)
    mask = (freq_thz >= BAND_MIN_THZ) & (freq_thz <= BAND_MAX_THZ)
    if np.count_nonzero(mask) < 8:
        raise HTTPException(
            status_code=400,
            detail=(
                f"After converting the first column to frequency, fewer than 8 points fall inside "
                f"the {BAND_MIN_THZ:.0f}–{BAND_MAX_THZ:.0f} THz band."
            ),
        )

    freq_crop = freq_thz[mask]
    spectrum_crop = y[mask]

    order = np.argsort(freq_crop)
    freq_crop = freq_crop[order]
    spectrum_crop = spectrum_crop[order]

    spectrum_norm = spectrum_crop / (np.max(np.abs(spectrum_crop)) + 1e-8)
    resampler = interp1d(freq_crop, spectrum_norm, kind="linear", fill_value="extrapolate")
    freq_resampled = np.linspace(freq_crop.min(), freq_crop.max(), TARGET_POINTS)
    spectrum_resampled = resampler(freq_resampled).astype(np.float32)

    return freq_resampled, spectrum_resampled


# -----------------------------
# Inference
# -----------------------------
def run_inversion_from_bytes(contents: bytes) -> dict:
    df = _read_csv_bytes(contents)
    freq_resampled, spectrum_resampled = preprocess_spectrum(df)

    model = load_model()
    model_input = torch.from_numpy(spectrum_resampled).view(1, 1, -1).to(DEVICE)

    with torch.no_grad():
        pred = model(model_input).squeeze().cpu().numpy().astype(np.float64)

    pred_nonneg = np.clip(pred, 0.0, None)
    pred_display = pred_nonneg / (pred_nonneg.max() + 1e-8)
    z_um = np.arange(TARGET_POINTS, dtype=np.float64) * DX_UM

    return {
        "model_name": model.__class__.__name__,
        "raw_points": int(len(df)),
        "band_min_thz": BAND_MIN_THZ,
        "band_max_thz": BAND_MAX_THZ,
        "resampled_points": TARGET_POINTS,
        "profile_min": float(pred_nonneg.min()),
        "profile_max": float(pred_nonneg.max()),
        "spectrum_thz": freq_resampled.astype(float).tolist(),
        "spectrum_norm": spectrum_resampled.astype(float).tolist(),
        "z_um": z_um.tolist(),
        "profile_raw": pred_nonneg.astype(float).tolist(),
        "profile_norm": pred_display.astype(float).tolist(),
    }


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="SpectraLab Local Inversion API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    model = load_model()
    return {
        "status": "ok",
        "model": model.__class__.__name__,
        "model_path": MODEL_PATH,
        "target_points": TARGET_POINTS,
        "band_min_thz": BAND_MIN_THZ,
        "band_max_thz": BAND_MAX_THZ,
    }


@app.post("/api/invert")
async def invert(file: UploadFile = File(...)) -> dict:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv spectrum file.")

    contents = await file.read()
    return run_inversion_from_bytes(contents)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
