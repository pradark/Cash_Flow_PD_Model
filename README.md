# Bank Transaction Default (PD) Model

**Nova Credit | Cash Flow Data Science**
**Date:** April 2026 | **Version:** 1.0

---

## Overview

An explainable binary classification model predicting customer **probability of default (PD)** from bank transaction (cash flow) data. Label = 1 indicates default.

| Metric | Train | Test | Gap |
|---|---|---|---|
| AUC | 0.8392 | 0.8252 | 1.41% |
| KS | 0.5321 | 0.5192 | 1.29% |

Gap target: < 2% (anti-overfitting constraint applied during Bayesian optimisation).

---

## Methodology

The pipeline follows the same methodology as the [Restaurant PD Model](https://github.com/pradark/Restaurant-PD-model):

1. **NLP Feature Extraction** — Rule-based extractor (130+ regex patterns) derives 7 structured features from free-text transaction descriptions (`feature_0`)
2. **WoE / IV Analysis** — Custom Weight of Evidence / Information Value implementation. Features with IV >= 1.0 excluded as potential leakage
3. **SHAP RFE** — `probatus.ShapRFECV` with 5-fold CV, step=0.25, selects optimal feature count at elbow of CV AUC curve
4. **LightGBM** — Gradient boosted trees with Bayesian hyperparameter optimisation. Anti-overfitting penalty: `score = val_AUC - 5 * max(0, gap - 0.02)`
5. **Evaluation** — ROC AUC, KS statistic, SHAP feature importance, decile expected vs actual

---

## Dataset

- **Records:** 6,000
- **Features:** 14 numeric + 1 free-text transaction description
- **Target:** `label` (1 = default, 0 = non-default)
- **Event rate:** 13.3% (6.5:1 class imbalance)
- **Split:** 75% train (4,500) / 25% test (1,500), stratified

---

## NLP Features Extracted from `feature_0`

| Feature | Type | Description |
|---|---|---|
| `merchant_category` | Categorical | 16-class merchant type (Payroll, Transfer_P2P, Grocery, etc.) |
| `txn_channel` | Categorical | Payment channel (Card, ACH, ATM, Wire, Mobile, Check) |
| `txn_direction` | Categorical | Debit / Credit / Unknown |
| `is_recurring` | Binary | 1 if tagged RECURRING |
| `is_p2p` | Binary | 1 if P2P transfer (Venmo, Zelle, CashApp) |
| `is_international` | Binary | 1 if international transaction |
| `merchant_risk_tier` | Categorical | High / Medium / Low risk tier |

---

## Final Model Features (5)

Selected by SHAP RFE (CV AUC 0.8239 at n=5, elbow point):

| Feature | IV | Strength |
|---|---|---|
| feature_7 | 0.7355 | Very Strong |
| feature_3 | 0.6323 | Very Strong |
| feature_6 | 0.3523 | Strong |
| feature_5 | 0.3255 | Strong |
| feature_1 | 0.2301 | Medium |

---

## Repository Contents

```
Bank_Transaction_Default_Model.ipynb   # Main analysis notebook (54 cells)
Research_Paper_v2.pdf                  # Full research paper (18 pages)
build_pdf.py                           # ReportLab PDF generation script
iv_table.csv                           # IV rankings for all 21 features
table_decile.csv                       # Decile-level expected vs actual
fig_woe_pdp_a.png                      # WoE PDP plots (features 1-3)
fig_woe_pdp_b.png                      # WoE PDP plots (features 4-5)
fig_decile.png                         # Decile expected vs actual chart
fig3_shap_rfe.png                      # SHAP RFE elbow curve
fig4_roc.png                           # ROC curve (train vs test)
fig5_ks.png                            # KS statistic plot
fig6_confusion.png                     # Confusion matrix
fig7_shap.png                          # SHAP summary plot
fig2_eda.png                           # EDA overview
```

---

## Quick Start

```bash
pip install lightgbm probatus bayesian-optimization shap pandas numpy matplotlib seaborn scikit-learn

jupyter notebook Bank_Transaction_Default_Model.ipynb
```

> **Note:** `optbinning` is not compatible with Python 3.13 / ARM Mac. A custom WoE/IV implementation is used instead.

---

## Key Results: Decile Expected vs Actual (Test Set)

| Decile | N | Actual Defaults | Actual Rate | Expected Rate |
|---|---|---|---|---|
| 1 (highest risk) | 150 | 77 | 51.3% | 41.7% |
| 2 | 146 | 35 | 24.0% | 25.4% |
| 3 | 152 | 31 | 20.4% | 21.7% |
| 4 | 152 | 25 | 16.4% | 17.0% |
| 5 | 150 | 13 | 8.7% | 11.3% |
| 6 | 149 | 8 | 5.4% | 6.7% |
| 7 | 148 | 3 | 2.0% | 4.9% |
| 8 | 153 | 2 | 1.3% | 3.2% |
| 9 | 150 | 4 | 2.7% | 2.0% |
| 10 | 150 | 1 | 0.7% | 0.8% |

---

## Related Work

- [Restaurant PD Model](https://github.com/pradark/Restaurant-PD-model) — same methodology applied to restaurant industry default prediction
