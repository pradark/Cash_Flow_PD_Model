# Agent Prompt: Build a Bank Transaction PD (Default Detection) Model

Use this prompt to spin up an agent that builds an end-to-end binary default classification pipeline from a raw CSV of labelled bank transactions.

---

## Prompt

You are a credit risk data scientist. Build an end-to-end binary default detection model using the dataset at `{DATA_PATH}`.

### Dataset assumptions
- Target column: `{TARGET_COL}` (1 = default, 0 = non-default)
- Free-text transaction description column: `{TEXT_COL}` (may be absent — skip NLP steps if so)
- Numeric/categorical feature columns: all remaining columns except the target
- Output directory: `{OUTPUT_DIR}`

### Steps to execute in order

**1. Data profiling**
- Load the CSV. Print shape, dtypes, missing value counts, and target distribution.
- Identify sentinel values (e.g. 9,999,997) and replace with NaN.
- Fix mixed-type columns (string "0"/"1" mixed with numeric).
- Print the class imbalance ratio.

**2. NLP feature extraction (skip if no text column)**
- Use `TransactionNLPExtractor` from `skills/nlp_txn_extractor.py`.
- Extract 7 features: `merchant_category`, `txn_channel`, `txn_direction`, `is_recurring`, `is_p2p`, `is_international`, `merchant_risk_tier`.
- Run `extractor.transform(df, col='{TEXT_COL}')`.

**3. Train/test split**
- Stratified 75/25 split, `random_state=42`.
- Confirm default rate is preserved in both splits (within 0.5 pp).

**4. WoE / IV analysis**
- Use `compute_all_iv` from `skills/woe_iv.py`.
- Separate numeric columns (use `compute_woe_iv_numeric`) from categorical (use `compute_woe_iv_categorical`).
- Fit bins on TRAINING data only.
- Exclude features with IV > 1.0 (suspected leakage) and IV < 0.02 (useless).
- Print IV table sorted descending; label strength.
- Plot WoE PDPs for all surviving features using `plot_all_woe_pdps`.

**5. WoE encoding**
- Apply `apply_woe(df_train, bins)` and `apply_woe(df_test, bins)` (no refitting on test).
- Drop original columns; use only `*_woe` encoded versions for modelling.

**6. SHAP Recursive Feature Elimination**
- Use `probatus.ShapRFECV` with a `LGBMClassifier(n_estimators=100, random_state=42)`.
- Parameters: `step=0.25, cv=5, scoring='roc_auc'`.
- Fit on training WoE features.
- Plot the RFE curve (n_features vs CV AUC).
- Select features at the elbow (max CV AUC, preferring fewer features when AUC difference < 0.002).
- Print selected feature list.

**7. LightGBM with Bayesian optimisation**
- Use `BayesLGBM` from `skills/bayes_lgbm.py`.
- Parameters: `gap_threshold=0.02, gap_penalty=5.0, n_init=10, n_iter=40, cv=5`.
- Fit on training set with selected features only.
- Print best hyperparameters.

**8. Model evaluation**
- Use `performance_metrics` and `plot_auc2` from `skills/model_utils.py`.
- Compute and print: Train AUC, Test AUC, Train KS, Test KS, AUC gap, KS gap.
- Flag if any gap exceeds 2 pp.
- Plot: ROC curves (train + test overlay), KS plots, confusion matrix at threshold 0.50.
- Plot SHAP summary (beeswarm) on test set.
- Compute decile table with `exp_vs_act`; plot with `plot_decile_chart`.

**9. Persist outputs**
Save to `{OUTPUT_DIR}`:
- `final_model.pkl` — dict with keys: `model`, `best_params`, `X_tr`, `X_te`, `y_tr`, `y_te`, `tr_auc`, `te_auc`, `tr_ks`, `te_ks`, `tr_proba`, `te_proba`
- `woe_bins.pkl` — dict with keys `numeric` and `categorical`, each mapping feature → bins_df
- All figures as PNG files prefixed `fig_`
- `table_decile.csv` — decile validation table

**10. Print final summary**
```
Dataset : {n} rows, {n_defaults} defaults ({pct:.1f}%)
Features selected : {selected_features}
Train AUC : {tr_auc:.4f}   Train KS : {tr_ks:.4f}
Test  AUC : {te_auc:.4f}   Test  KS : {te_ks:.4f}
AUC gap   : {gap:.4f} pp   {'✓ within 2pp' if gap < 0.02 else '✗ EXCEEDS TARGET'}
```

### Guard rails
- Never fit any transformer (WoE bins, imputers) on test data.
- If class imbalance ratio > 10:1, note it and consider `scale_pos_weight` in LightGBM.
- If any IV > 1.0 feature looks suspicious, print a warning but do not make a final leakage determination — flag for domain review.
- Do not use optbinning on Python 3.13+ (ortools dependency conflict). Use `skills/woe_iv.py` instead.

### Required skill files
Place these in `skills/` alongside your notebook:
- `skills/woe_iv.py`
- `skills/model_utils.py`
- `skills/bayes_lgbm.py`
- `skills/nlp_txn_extractor.py`

### Required packages
```
lightgbm>=4.0
scikit-learn>=1.5
shap>=0.44
probatus>=3.1
bayesian-optimization>=1.5
pandas>=2.0
numpy>=1.26
scipy>=1.10
matplotlib>=3.8
seaborn>=0.12
```
