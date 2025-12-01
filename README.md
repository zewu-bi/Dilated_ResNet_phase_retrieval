Dilated ResNet Phase Retrieval

1D Dilated ResNet for Coherent Transition Radiation (CTR) Phase Retrieval
From form factor → longitudinal beam profile using a physics-informed deep neural network.

This repository provides a full pipeline for simulating electron-beam longitudinal structures, computing CTR form factors, training a Dilated ResNet for phase retrieval, and evaluating reconstruction performance.

The project is designed for high-resolution (1024-point) spectra, two-Gaussian (G2) beams, and supports customizable data distributions (uniform/log-uniform).

✨ Features

Physics-informed data generation

Multi-Gaussian electron beams

Log-uniform charge ratio sampling

Optional log-uniform sigma / peak distance sampling

Form factor calculation

Fourier-based analytical calculation

Supports noisy spectrum for robustness training

Dilated ResNet

1D CNN with exponentially increasing receptive field

Residual blocks

Suitable for band-limited, global-dependency inverse problems

Training & evaluation pipeline

GPU-accelerated training

Automatic saving of best checkpoints

Experiment logs & visualization
