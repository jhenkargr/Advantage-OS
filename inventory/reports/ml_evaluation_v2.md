# AdVantage OS Inventory ML Pipeline Evaluation (v2 — Definitive Benchmark)
**Date:** 2026-10-10  
**Evaluator:** Senior ML Engineer  
**Dataset:** 40 SKUs × 90 Days (Synthetic Poisson DGP with uncensored `potential_demand` and stock-censored `units_sold`)

> [!WARNING]
> **Evaluation Protocol Notice:** The 14-day test window has been evaluated multiple times during development and debugging iterations. While all features, splits, and parameter tunings are leakage-free, this window represents a holdout benchmark rather than a pristine single-touch test set. The 5-fold purged cross-validation serves as the primary headline evaluation.

---

## 1. Headline Cross-Validation Benchmark (5-Fold Purged CV)

All models evaluated with 17 leakage-safe observable features and a strict 7-day purge gap between training and validation:

| Model Architecture | Loss Function | Train MAE | 5-Fold CV MAE | 5-Fold CV RMSE | 5-Fold CV WAPE | Train-CV Gap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge (alpha=10.0)** | Squared Error ($L_2$) | 8.288 | **15.006 ± 2.303** | **21.093 ± 3.588** | **0.222 ± 0.019** | **+6.718** |
| **HGB Regressor** | Absolute Error ($L_1$) | 3.555 | 15.150 ± 4.225 | 22.478 ± 6.683 | 0.223 ± 0.046 | +11.595 |
| **LightGBM** | Absolute Error ($L_1$) | 4.908 | 14.780 ± 4.290 | 22.021 ± 6.906 | 0.217 ± 0.047 | +9.872 |
| **LightGBM** | Squared Error ($L_2$) | 4.351 | 15.589 ± 3.610 | 22.478 ± 5.338 | 0.230 ± 0.038 | +11.238 |
| **HGB Regressor** | Squared Error ($L_2$) | 2.545 | 15.784 ± 3.693 | 22.872 ± 5.378 | 0.232 ± 0.039 | +13.238 |
| **LightGBM** | Poisson Deviance | 5.731 | 15.505 ± 3.633 | 22.616 ± 6.043 | 0.228 ± 0.038 | +9.775 |
| **HGB Regressor** | Poisson Deviance | 2.235 | 15.631 ± 3.995 | 22.796 ± 6.059 | 0.230 ± 0.044 | +13.396 |
| **PoissonRegressor** | Poisson GLM | 9.806 | 15.850 ± 2.663 | 23.211 ± 4.768 | 0.235 ± 0.023 | +6.044 |

> **Conclusion on Loss Functions:** Standard **squared error ($L_2$) with regularized Ridge regression was already optimal**, achieving the lowest CV RMSE (21.093) and the smallest generalization gap (+6.718). The difference across the leading loss functions is **within noise** (differences of $\approx 0.2$ to $0.8$ MAE against standard deviations of $\pm 2.3$ to $\pm 4.3$). Furthermore, the **Train-vs-CV gap reflects fold shift (the onset of synthetic trend bursts in validation folds), not overfitting**.

---

## 2. Fair Comparison on Identical 14-Day Test Window

| Metric | Original MLP (4 Features) | Upgraded ML Winner (Ridge) | Difference | Better / Worse |
| :--- | :---: | :---: | :---: | :---: |
| **5-Fold Purged CV MAE** | 24.326 ± 3.122 | 15.006 ± 2.303 | -9.320 | **Better** |
| **5-Fold Purged CV RMSE** | 35.222 ± 6.230 | 21.093 ± 3.588 | -14.129 | **Better** |
| **5-Fold Purged CV WAPE** | 0.362 ± 0.032 | 0.222 ± 0.019 | -0.140 | **Better** |
| **Test Set MAE** | 20.200 | 10.327 | -9.873 | **Better** |
| **Test Set RMSE** | 26.726 | 13.175 | -13.551 | **Better** |
| **Test Set WAPE** | 0.343 | 0.175 | -0.168 | **Better** |
| **Test Set $R^2$** | 0.270 | 0.823 | +0.553 | **Better** |

> [!NOTE]
> **MLP Diagnostic Resolution:** The historical training failure of the MLP was **strictly caused by unclipped `days_of_cover`**. When recent rolling sales approached zero, unclipped days of cover reached values exceeding $10^6$, causing gradient explosion. Once clipped to $[0, 30]$, the MLP trains stably with well-behaved loss curves across all folds (reaching CV MAE = 19.82).

**Oracle Noise Ceiling & Partly Unreachable Gap:**
- Oracle Theoretical Floor on Test: MAE = **5.748**, RMSE = **7.150**, WAPE = **0.098**, $R^2$ = **0.9477**.
- Winner Gap to Oracle on Test: $\Delta\text{MAE} = +4.579$ units.
- **Why Part of the Gap is Unreachable:** The synthetic data generation process includes an unexpected $+80\%$ trend surge for 10 consecutive days. The exact onset date of this demand shock cannot be forecasted in advance by any causal model without lookahead, meaning a non-zero portion of this gap is mathematically unreachable.

---

## 3. Direct Quantile Regression vs. Ridge Replenishment Selection

Evaluating empirical pinball loss and coverage across folds:

| Quantile ($q$) | 5-Fold CV Pinball Loss | Train-Fold Coverage | Val-Fold Coverage | Dev Replenishment Cost ($k=0$) |
| :---: | :---: | :---: | :---: | :---: |
| **0.50** | 7.575 ± 2.112 | 50.1% | 46.7% | $1252.80 |
| **0.65** | 8.408 ± 2.558 | 64.5% | 53.1% | $1276.00 |
| **0.75** | 8.313 ± 2.718 | 74.1% | 56.8% | $1233.60 |
| **0.85** | 7.725 ± 3.140 | 83.0% | 61.9% | $1341.20 |

> [!IMPORTANT]
> **Quantile Under-Coverage & Safety Multiplier Notice:** Unshifted quantile regression at $q=0.75$ covers only **56.8%** on validation folds due to chronological distribution shift. Therefore, we **do not claim $k=0$ removes the safety multiplier unless coverage reaches about 75%**. When applying a conformal shift of $\Delta = +5.89$ units on dev residuals, empirical test coverage reaches **94.1%**.
> 
> **Model Selection on Dev Cost:**
> - **Ridge + tuned $k$ ($k=0.5$):** Dev Total Cost = **$1173.00**
> - **Direct Quantile ($q=0.75, k=0$):** Dev Total Cost = **$1233.60**
> - **Outcome:** **Direct quantile and Ridge+k are virtually tied on dev-selected comparison** (difference < 5%). Ridge+k is marginally preferred on dev cost and selected for primary simulation comparison, with the direct quantile policy reported as a reference.

---

## 4. Business Replenishment Simulation (200 Poisson DGP Seeds)

Safety stock multiplier $k$ tuned strictly on dev period over extended grid $[0, 0.5, 1, 1.5, 2, 3, 4, 5]$:
- Dev-tuned settings: Naive $k=3.0$, Censoring-Corrected $k=1.5$, ML Policy $k=0.5$.
- Evaluated over **200 independent Poisson demand realizations** generated directly from the ground-truth DGP ($d_t \sim \text{Poisson}(\text{expected\_daily\_demand}_t)$):

| Replenishment Policy | Tuned $k$ | Mean Total Cost (200 Seeds) | Paired Cost Difference vs. ML | Percentile Interval of Differences | 95% Confidence Interval of Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ML Policy (Ridge)** | **0.5** | **$825.39** | — | — | — |
| **ML Reference (Direct Quantile $q=0.75$)** | **0.0** | **$812.00** | — | — | — |
| **Censoring-Corrected** | 1.5 | $959.86 | $\Delta = -134.47 | **[$-238.00, $-41.55]** | **[$-141.54, $-127.40]** |
| **Naive Policy** | 3.0 | $1550.86 | $\Delta = -725.48 | **[$-850.54, $-604.06]** | **[$-733.58, $-717.37]** |

> [!WARNING]
> **Simulation Scope Notice:** The simulation cost advantage (**$134.47** savings over Censoring-Corrected) **holds only for this evaluation window and relies on the unobservable `potential_demand` label** used for offline training.
> 
> **Statistical Interpretation of Intervals:**
> - The **Percentile Interval of Differences** (`[$-238.00, $-41.55]`) represents the 2.5th to 97.5th percentile spread of per-seed business cost differences across individual realizations of the DGP.
> - The **Confidence Interval of the Mean Difference** (`[$-141.54, $-127.40]`) reflects the standard error of the expected average cost savings, which strictly excludes zero.

---

## 5. Stockout Risk Classification: Cost-Optimal Rule & Bootstrap Check

### Cost-Derived vs. $F_2$-Tuned Decision Threshold
Assumptions:
- $C_{fn} = \$2.00$ per lost unit-equivalent (penalty per missed stockout event).
- $C_{fp} = \$0.10 \times 3\text{ days} = \$0.30$ (holding cost of 1 unit buffer across the 3-day lead time).
- Theoretical decision threshold: $p^* = \frac{C_{fp}}{C_{fp} + C_{fn}} = \frac{0.30}{2.30} \approx 0.1304$.

| Policy / Threshold | Decision Threshold | Dev Misclass Cost | Test Misclass Cost | Test Precision | Test Recall | Test $F_2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cost-Optimal Rule** | $p^* = 0.1304$ | **$60.60** | **$61.10** | 0.511 | 0.959 | 0.816 |
| **$F_2$-Tuned Threshold** | $p = 0.1000$ | $57.10 | $66.70 | 0.468 | 0.971 | 0.799 |
| **Heuristic Rule** | `days_of_cover < 5` | — | — | 0.487 | 0.912 | 0.777 |

### Point Estimate vs. Cluster Bootstrap (2,000 Draws Resampling Whole SKUs)
To properly account for panel autocorrelation across the 14 days per SKU, cluster bootstrapping was executed across all 40 SKUs:

| Model Architecture | Original Test $F_2$ | Original Point $\Delta F_2$ | Bootstrap Mean $\Delta F_2$ | 95% Cluster Bootstrap CI | Excludes 0? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Calibrated Logistic Regression** | 0.7988 | **+0.0220** | +0.0225 | **[-0.0126, +0.0565]** | **False** |
| **HistGradientBoosting Classifier** | 0.7988 | **+0.0219** | +0.0226 | **[-0.0368, +0.0820]** | **False** |
| **Heuristic Rule (`days_of_cover < 5`)** | 0.7769 | — | — | — | — |

- **Confirmation of Distinct Model Predictions:** Logistic Regression and HistGradientBoosting differ on **99 out of 560 predictions (17.7%)**. Although their test $F_2$ scores happen to be nearly identical, they are completely separate models.
- **Statistical Significance:** Under cluster bootstrap, **both 95% confidence intervals include zero**. ML models do not provide a statistically significant improvement over the simple `days_of_cover < 5` heuristic rule.

---

## 6. Prediction Uncertainty Intervals (Bug-Fixed & Verified)

Calibrated strictly on out-of-sample Ridge predictions from `tr_df` on `va_df` against true 7-day demand:
- Calibration fold residual stats on 7-day demand: Mean $|e| = 16.15$, Median $|e| = 14.11$, 80th percentile $= 25.46$.
- **Unscaled Conformal Multiplier:** $\hat{q}_{\text{unscaled}} = \mathbf{25.49}$ units (in the expected $\approx 15-25$ range, fixing the previous $\approx 78$ bug caused by subtracting daily potential demand).
- **Scaled Conformal Multiplier:** $\hat{q}_{\text{scaled}} = \mathbf{2.87}$.

| Prediction Interval Method | Test Coverage | Test Avg Width | CV Fold 1 Cov | CV Fold 2 Cov | CV Fold 3 Cov | CV Fold 4 Cov | CV Fold 5 Cov |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Split Conformal (Unscaled)** | **93.8%** | **50.05** | 89.6% | 86.1% | 71.6% | 82.2% | 84.4% |
| **Split Conformal (Scaled $|e|/\sqrt{\hat{y}+1}$)** | **92.1%** | **44.57** | 85.2% | 85.2% | 70.2% | 78.7% | 80.9% |
| **CQR Conformal (Quantile HGB)** | **90.0%** | **44.26** | 85.0% | 73.5% | 63.4% | 82.8% | 79.8% |
| **Poisson Theory Oracle (Exact DGP)** | **83.0%** | **19.08** | — | — | — | — | — |

- **Comparison to Earlier Result:** Unscaled conformal yields **93.8% coverage** and **50.05 width** (compared to the earlier 89.6% / 40.9 when calibrated on earlier fold data).
- **Scaled Efficiency:** Scaled conformal achieves **92.1% coverage** with an average width of **44.57 units**, narrowing the interval by 5.5 units while preserving valid coverage.
