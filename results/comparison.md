# MFFGCN Reproduction vs. Paper

All numbers are reported on the RadioMapSeer DPM test split (last 100 city maps, unseen during training).

## Table 1 - Reproduction vs. paper (MFFGCN)

| Setup | Paper NMSE | Ours NMSE | Δ NMSE | Paper RMSE | Ours RMSE | Δ RMSE |
|---|---|---|---|---|---|---|
| Setup 1 (uniform) | 0.0043 | 0.0770 | +1691.5% | 0.0150 | 0.0728 | +385.3% |
| Setup 2 (25% restricted) | 0.0045 | 0.1016 | +2157.9% | 0.0154 | 0.0836 | +442.9% |
| Setup 3 (10-50% restricted) | 0.0046 | 0.1334 | +2801.0% | 0.0155 | 0.0958 | +518.1% |

## Table 2 - Effect of the graph refinement stage (ours)

| Setup | Coarse NMSE | Refined NMSE | Gain | Coarse RMSE | Refined RMSE | Gain |
|---|---|---|---|---|---|---|
| Setup 1 (uniform) | 0.1146 | 0.0770 | -32.8% | 0.0888 | 0.0728 | -18.0% |
| Setup 2 (25% restricted) | 0.1318 | 0.1016 | -22.9% | 0.0952 | 0.0836 | -12.2% |
| Setup 3 (10-50% restricted) | 0.1240 | 0.1334 | +7.6% | 0.0924 | 0.0958 | +3.7% |

## Table 3 - Reconstruction quality by region (ours)

| Setup | Whole map RMSE | Constrained region RMSE | Observed pixels RMSE |
|---|---|---|---|
| Setup 1 (uniform) | 0.0728 | — | 0.0018 |
| Setup 2 (25% restricted) | 0.0836 | 0.0516 | 0.0025 |
| Setup 3 (10-50% restricted) | 0.0958 | 0.0662 | 0.0028 |

## Table 4 - Robustness to uneven sampling

| Setup | Paper RMSE increase vs S1 | Ours RMSE increase vs S1 |
|---|---|---|
| Setup 2 (25% restricted) | +2.7% | +14.8% |
| Setup 3 (10-50% restricted) | +3.3% | +31.6% |

## Table 5 - Where our model sits among published baselines

| Method | Setup 1 (NMSE / RMSE) | Setup 2 (NMSE / RMSE) | Setup 3 (NMSE / RMSE) |
|---|---|---|---|
| KNN (paper) | 0.9973 / 0.2312 | 0.9980 / 0.2313 | 0.9981 / 0.2313 |
| Kriging (paper) | 0.4545 / 0.1389 | 0.4545 / 0.1389 | 0.4575 / 0.1401 |
| AE (paper) | 0.0387 / 0.0456 | 0.0753 / 0.0627 | 0.0829 / 0.0657 |
| DeepAE (paper) | 0.0156 / 0.0288 | 0.0276 / 0.0319 | 0.0395 / 0.0422 |
| U-Net (paper) | 0.0063 / 0.0182 | 0.0075 / 0.0199 | 0.0081 / 0.0206 |
| RadioUNet (paper) | 0.0049 / 0.0160 | 0.0058 / 0.0175 | 0.0063 / 0.0181 |
| Pix2pix+ (paper) | 0.0046 / 0.0156 | 0.0053 / 0.0169 | 0.0054 / 0.0172 |
| RME-GAN (paper) | 0.0043 / 0.0151 | 0.0051 / 0.0165 | 0.0055 / 0.0172 |
| MFFGCN (paper) | 0.0043 / 0.0150 | 0.0045 / 0.0154 | 0.0046 / 0.0155 |
| **Ours (reproduced)** | 0.0770 / 0.0728 | 0.1016 / 0.0836 | 0.1334 / 0.0958 |
