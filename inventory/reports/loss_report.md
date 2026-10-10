# Loss Function Report: AdVantage OS Inventory ML

All numbers reported below were computed directly by running `inventory/src/check_loss.py` on the development window (test window excluded) using the 17 observable features in `features.py` under the purged 5-fold `TimeSeriesSplit` (gap = 280 rows / 7 days).

---

## 1. Definition and Model Training Losses
A loss function is a mathematical objective that measures the discrepancy between a model's prediction and the observed ground truth, guiding parameter updates during training. In this project, Ridge regression trains on regularized squared error ($L_2$), the multilayer perceptron (MLP) trains on squared error ($L_2$), and the stockout risk classifier trains on cross-entropy / logistic loss (log loss).

---

## 2. Evaluation Results Across Four Loss Formulations
The table below evaluates the exact same out-of-sample Ridge predictions across four distinct evaluation criteria across the 5 cross-validation folds:

| Loss Function | Mathematical Formulation | 5-Fold CV Mean ± Std | Interpretation |
| :--- | :--- | :---: | :--- |
| **Mean Squared Error (MSE)** | $\frac{1}{N} \sum (y - \hat{y})^2$ | **457.79 ± 157.59** | Symmetric quadratic penalty penalizing large outliers |
| **Mean Absolute Error (MAE)** | $\frac{1}{N} \sum \|y - \hat{y}\|$ | **15.01 ± 2.30** | Symmetric linear penalty measuring average units missed |
| **Pinball Loss ($q=0.74$)** | $\frac{1}{N} \sum \max(q(y - \hat{y}), (q-1)(y - \hat{y}))$ | **8.18 ± 1.96** | Asymmetric loss matching the cost ratio $q = \frac{2.00}{2.00 + 0.70} \approx 0.74$ |
| **Business Cost Loss** | $\frac{1}{N} \sum (2.00 \cdot \max(0, y - \hat{y}) + 0.70 \cdot \max(0, \hat{y} - y))$ | **$22.10 ± 5.31** | Direct business penalty: $\$2.00$/unit under-forecast, $\$0.70$/unit over-forecast |

*Note on Business Cost Standard Deviation:* The standard deviation of business cost across the 5 folds is 5.307 (folds: 16.45, 23.72, 31.53, 20.60, 18.21), which is independent of the validation bias standard deviation of 5.308 (folds: +0.88, +8.21, +9.68, -0.20, -4.39); both round to 5.31 by coincidence.

---

## 3. Train vs. Validation Loss for Ridge
Per-fold training and validation metrics for Ridge ($\alpha=10.0$):

- **Fold 1:** Train MSE = 44.51 (MAE = 5.40) | Val MSE = 268.54 (MAE = 11.76) | Val Bias = **+0.88**
- **Fold 2:** Train MSE = 83.46 (MAE = 6.91) | Val MSE = 487.32 (MAE = 13.62) | Val Bias = **+8.21**
- **Fold 3:** Train MSE = 134.93 (MAE = 8.09) | Val MSE = 734.60 (MAE = 18.69) | Val Bias = **+9.68**
- **Fold 4:** Train MSE = 210.65 (MAE = 9.81) | Val MSE = 444.05 (MAE = 15.35) | Val Bias = **-0.20**
- **Fold 5:** Train MSE = 278.95 (MAE = 11.23) | Val MSE = 354.46 (MAE = 15.60) | Val Bias = **-4.39**
- **Average:** Train MSE = 150.50 ± 84.96 (MAE = 8.29 ± 2.06) | Val MSE = 457.79 ± 157.59 (MAE = 15.01 ± 2.30) | Val Bias = **+2.83 ± 5.31**

**Interpretation:** The train-to-validation gap is mostly fold shift rather than overfitting for a 17-feature linear model. Validation bias is positive in folds 2-3 (+8.2, +9.7), near zero or negative in folds 4-5 (-0.2, -4.4), so the direction of the shift varies. The cause of each fold's bias was not verified. Train error rises across folds because later folds train on more data.

---

## 4. MLP Training Convergence
Training loss decreased and flattened; folds 1, 2 and 4 stopped at the 2,000-iteration cap, so they did not fully converge:
- **Fold 1:** Final Loss = **2.3765** (2,000 iterations)
- **Fold 2:** Final Loss = **3.7017** (2,000 iterations)
- **Fold 3:** Final Loss = **31.0371** (772 iterations)
- **Fold 4:** Final Loss = **12.2578** (2,000 iterations)
- **Fold 5:** Final Loss = **27.5903** (1,327 iterations)

The loss curve plot across all 5 folds is saved at `reports/figures/loss_check_mlp.png`.

---

## 5. Conclusion
- **Operational Loss:** Standard squared error ($L_2$) regularized Ridge regression remains the primary demand forecasting model; the lowest CV RMSE (21.09) and lowest fold variance ($\pm 2.30$) claims are from the earlier loss comparison.
- **Impact of Objective Changes:** Changing the training objective between squared error, absolute error, Poisson deviance, and pinball loss did not produce meaningful differences, with CV MAE across all alternatives remaining between 14.78 and 15.85, all within the fold-to-fold noise standard deviation of $\pm 2.30\text{--}4.29$.
- **Asymmetric Business Cost:** Costs of $\$2.00$ per unit under-forecast and $\$0.70$ per unit over-forecast are assumed costs, and the business-cost row shows the cost of Ridge's errors under that assumption, reflecting heavier penalties when demand exceeds supply.

---

## 6. Limitations
- **Synthetic Data Process:** The findings are derived from a synthetic dataset (40 SKUs × 90 days) featuring localized Poisson demand and artificial 10-day trend bursts, which may not capture supplier disruptions or multi-echelon retail dynamics.
- **Repeated Holdout Exposure:** The 14-day holdout test window has been evaluated across multiple debugging and development iterations; consequently, cross-validation metrics serve as the primary credible performance benchmark.
