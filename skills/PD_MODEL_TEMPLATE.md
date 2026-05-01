# PD Model Agent Template
> Copy this file and the `skills/` folder to any new project.
> Fill in the CONFIG block, then feed this file to an agent as the opening prompt.

---

## CONFIG — fill these in before handing to an agent

```
DATA_PATH        = path/to/your_dataset.csv
TARGET_COL       = label          # binary: 1=default/event, 0=non-default/non-event
TEXT_COLS        = []             # list of free-text columns to extract NLP features from; [] if none
OUTPUT_DIR       = ./             # where to save models, figures, tables
AUTHOR           = Your Name
PROJECT_NAME     = Your_Project_Name
```

---

## Agent Prompt

You are a credit risk data scientist. Build an end-to-end binary probability of default (PD) model using the dataset and config above. Follow the steps below in order. Use the skill files in `skills/` for all core logic — do not reimplement what is already there.

---

### Step 1 — Data Profiling

- Load `DATA_PATH`. Print: shape, dtypes, missing value counts per column, and target distribution.
- Identify and replace sentinel values (e.g. 9,999,997 → NaN; "N/A", "unknown" → NaN).
- Fix mixed-type columns (e.g. string "0"/"1" mixed with numeric → cast to numeric).
- Print the class imbalance ratio (non-events : events).
- Flag any column with >20% missing values for review.
- Classify each column as: Identifier, Numeric, Categorical, Binary, Text, or Date.

---

### Step 2 — NLP Feature Extraction *(skip if TEXT_COLS is empty)*

For each column in `TEXT_COLS`, extract structured features using `TransactionNLPExtractor` from `skills/nlp_txn_extractor.py`, or write custom regex rules if the text domain differs from financial transactions:

```python
from skills.nlp_txn_extractor import TransactionNLPExtractor
extractor = TransactionNLPExtractor()
df = extractor.transform(df, col=text_col)
```

Extend or replace the default pattern dictionaries via `extra_merchants={}` to match the domain (e.g. social media signals, bureau codes, marketing categories). All extracted features are treated as regular columns in subsequent steps.

---

### Step 3 — Train / Test Split

```python
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                            stratify=y, random_state=42)
```

Confirm the event rate is preserved in both splits (within 0.5 pp). If a time column is available, use a chronological split instead (earlier dates → train, later → test). Fit all transformers on training data only — never on test.

---

### Step 4 — WoE / IV Analysis

Use `skills/woe_iv.py`:

```python
from skills.woe_iv import compute_all_iv, apply_woe, plot_all_woe_pdps

iv_df, bins = compute_all_iv(df_train, target_col=TARGET_COL,
                              numeric_cols=num_feats,
                              categorical_cols=cat_feats)
```

- Print IV table sorted descending with strength labels.
- **Exclude** features with IV > 1.0 (suspected leakage — flag for domain review) and IV < 0.02 (no signal).
- Apply WoE encoding to train and test: `apply_woe(df_train, bins)`, `apply_woe(df_test, bins)`.
- Plot WoE PDPs for surviving features: `plot_all_woe_pdps(bins, selected_features)`.
- Save `iv_df` as `{OUTPUT_DIR}/data/iv_table.csv`.

---

### Step 5 — SHAP Recursive Feature Elimination

```python
from probatus.feature_elimination import ShapRFECV
from lightgbm import LGBMClassifier

shap_rfe = ShapRFECV(LGBMClassifier(n_estimators=100, random_state=42),
                     step=0.25, cv=5, scoring='roc_auc', n_jobs=-1)
shap_rfe.fit_compute(X_tr_woe, y_tr)
shap_rfe.plot()
selected_features = shap_rfe.get_reduced_features_set(num_features='best')
```

Select features at the elbow (max CV AUC; prefer fewer features when the AUC difference is < 0.002). Print the final feature list.

---

### Step 6 — Model Training with Bayesian Optimisation

Use `BayesLGBM` from `skills/bayes_lgbm.py`:

```python
from skills.bayes_lgbm import BayesLGBM

model = BayesLGBM(gap_threshold=0.02, gap_penalty=5.0,
                   n_init=10, n_iter=40, cv=5)
model.fit(X_tr_woe[selected_features], y_tr)
print(model.best_params_)
```

The objective penalises train/val AUC gaps above 2 pp to prevent overfitting.

---

### Step 7 — Evaluation

Use `skills/model_utils.py`:

```python
from skills.model_utils import performance_metrics, gap_check, plot_auc2, \
                                plot_confusion, exp_vs_act, plot_decile_chart

tr_proba = model.predict_proba(X_tr_woe[selected_features])[:, 1]
te_proba = model.predict_proba(X_te_woe[selected_features])[:, 1]

tr_auc, tr_ks = performance_metrics(y_tr, tr_proba, label='Train')
te_auc, te_ks = performance_metrics(y_te, te_proba, label='Test')
gap_check(tr_auc, te_auc, tr_ks, te_ks)

plot_auc2(y_tr, tr_proba, y_te, te_proba, title=PROJECT_NAME,
          save_path=f'{OUTPUT_DIR}/figures/fig_roc_ks.png')
plot_confusion(y_te, (te_proba >= 0.5).astype(int),
               save_path=f'{OUTPUT_DIR}/figures/fig_confusion.png')

tbl = exp_vs_act(y_te, te_proba)
plot_decile_chart(tbl, save_path=f'{OUTPUT_DIR}/figures/fig_decile.png')
tbl.to_csv(f'{OUTPUT_DIR}/data/table_decile.csv', index=False)
```

Also plot SHAP summary:

```python
import shap
explainer = shap.TreeExplainer(model.model_)
shap_vals = explainer.shap_values(X_te_woe[selected_features])
shap.summary_plot(shap_vals, X_te_woe[selected_features], show=False)
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap.png', bbox_inches='tight', dpi=120)
```

---

### Step 8 — Persist Outputs

```python
import pickle

with open(f'{OUTPUT_DIR}/final_model.pkl', 'wb') as f:
    pickle.dump({
        'model':       model.model_,
        'best_params': model.best_params_,
        'features':    selected_features,
        'woe_bins':    bins,
        'X_tr': X_tr_woe[selected_features], 'X_te': X_te_woe[selected_features],
        'y_tr': y_tr, 'y_te': y_te,
        'tr_auc': tr_auc, 'te_auc': te_auc,
        'tr_ks':  tr_ks,  'te_ks':  te_ks,
        'tr_proba': tr_proba, 'te_proba': te_proba,
    }, f)
```

---

### Step 9 — Print Final Summary

```
=====================================
PROJECT  : {PROJECT_NAME}
AUTHOR   : {AUTHOR}
=====================================
Dataset  : {n} rows | {n_events} events ({pct:.1f}%) | ratio {ratio:.1f}:1
Features : {selected_features}
-------------------------------------
           AUC       KS
Train    : {tr_auc:.4f}   {tr_ks:.4f}
Test     : {te_auc:.4f}   {te_ks:.4f}
Gap      : {auc_gap:.4f}   {ks_gap:.4f}   {'✓ within 2pp' if auc_gap < 0.02 else '✗ EXCEEDS TARGET'}
-------------------------------------
Top decile captures {top_decile_pct:.1f}% of events
=====================================
```

---

## Guard Rails

- Never refit WoE bins or any transformer on test data.
- IV > 1.0 → flag as suspected leakage; do not auto-exclude without domain review.
- If imbalance ratio > 10:1, evaluate `scale_pos_weight` in LightGBM.
- If a time dimension exists, prefer chronological splitting over random splitting.
- Decision threshold of 0.50 is a starting point — calibrate against a cost matrix for production.
- Do not use `optbinning` on Python 3.13+ (ortools conflict). Use `skills/woe_iv.py` instead.

---

## Skill Files

| File | Purpose |
|------|---------|
| `skills/woe_iv.py` | WoE/IV computation, encoding, PDP plots |
| `skills/model_utils.py` | performance_metrics, plot_auc2, exp_vs_act, plot_decile_chart |
| `skills/bayes_lgbm.py` | BayesLGBM with anti-overfitting Bayesian optimisation |
| `skills/nlp_txn_extractor.py` | Configurable rule-based NLP feature extractor for text columns |

## Required Packages

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
