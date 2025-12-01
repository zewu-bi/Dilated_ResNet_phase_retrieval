# Experiment Logs

| Exp ID | Dataset | Batch Size | Model | Depth | Learning Rate | Epochs | Notes |
|-------|---------|------------|--------|--------|----------------|---------|--------|
| 1 | G2_100_800_1_5_20_40_1000 | 32 | 1D CNN | 3 | 1e-3 | 2000 | Slow convergence; predictions collapse to a flat line. |
| 2 | G2_100_800_1_5_20_40_1000 | 32 | Dilated ResNet | 6 | 1e-3 | 200 | Basically works. |
| 3 | G2_100_800_1_5_20_40_1000 | 32 | Dilated ResNet | 6 | 1e-3 | 500 | Validation loss plateaus after epoch 180; model saved. |
| 4 | G2_100_800_1_5_20_40_1000 | 32 | Dilated ResNet | 6 | 1e-3 | 200 | Same parameters as Exp 2. |
| 5 | G2_100_800_1_5_20_40_10000 | 32 | Dilated ResNet | 6 | 1e-3 | 200 | Training becomes slow; common patterns collapse; rare cases generalize poorly. |
| 6 | G2R_100_800_1_5_20_40_1_3_10000 | 256 | Dilated ResNet | 6 | 1e-3 | 200 | Good generalization for large height differences; poor for width variations. |
| 7 | G2RR_100_800_1_5_20_40_1_3_03_3_10000 | 256 | Dilated ResNet | 6 | 1e-3 | 600 | sigma_ratio is physically unrealistic; overall training worse than Exp 6; extreme sigma distributions generalize poorly. |
| 8 | G2RP_100_800_2_5_1_3_20_40_05_3_30000 | 256 | Dilated ResNet | 6 | 1e-3 | 200 | Q_ratio + sigma_prior + 30000 samples; rare-case generalization still unsatisfactory. |
