# 1D Dilated ResNet for Coherent Transition Radiation (CTR) Phase Retrieval

From form factor → longitudinal beam profile using a physics-informed deep neural network.

This repository provides a full pipeline for simulating electron-beam longitudinal structures, computing CTR form factors, training a Dilated ResNet for phase retrieval, and evaluating reconstruction performance.

The project is designed for high-resolution (1024-point) spectra, two-Gaussian (G2) beams, and supports customizable data distributions (uniform/log-uniform).

✨ Features

Physics-informed data generation;
Multi-Gaussian electron beams;
Log-uniform charge ratio sampling;
Fourier-based analytical form factor calculation;
Dilated ResNet with exponentially increasing receptive field;
Residual blocks;
Suitable for band-limited, global-dependency inverse problems;
Training & evaluation pipeline;
GPU-accelerated training;
Experiment logs & visualization;

📂 Directory Structure

project_root/
│
├── beam_profile_library/           # Library of ground-truth electron bunches
├── calculated_form_factor/         # Precomputed form factors (FFT or analytical)
├── dataset/                        # Final paired dataset: (form_factor, beam_profile)
├── generated_beam_profile/         # Synthetic beam profiles from generator
├── logs/                           # TensorBoard logs, training curves
├── model/                          # Saved models (best checkpoints)
│
├── 1D_Dilated_ResNet.ipynb         # Model definition, trainting and testing
├── beam_generator.ipynb            # G2 beam generator (charge ratio, sigma, distance)
├── form_factor_calculator.ipynb    # FFT-based (or analytical) |F(ω)|² computation
├── experiment_logs.ipynb           
│
└── README.md


