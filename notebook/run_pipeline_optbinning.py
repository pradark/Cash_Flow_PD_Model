"""
Cash Flow PD Model — End-to-End Pipeline (optbinning + LightGBM)
Run with: /opt/anaconda3/envs/optbinning_env/bin/python3 run_pipeline_optbinning.py
"""

import sys, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              precision_score, recall_score, f1_score)
import lightgbm as lgb
import shap
from bayes_opt import BayesianOptimization
from optbinning import OptimalBinning

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────

BASE       = os.path.dirname(os.path.abspath(__file__)) + '/..'
DATA_PATH  = os.path.dirname(os.path.abspath(__file__)) + '/../../data/20230400_Cash_DS.csv'
FIG_DIR    = f'{BASE}/figures'
DATA_DIR   = f'{BASE}/data'
SEED       = 42
TARGET     = 'label'
PROJECT    = 'Cash Flow PD Model'
AUTHOR     = 'Pradeep Arkachar'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 120,
                     'axes.spines.top': False, 'axes.spines.right': False})

# ── 1. Load & Profile ─────────────────────────────────────────────────────────

print("=" * 60)
print(f"  {PROJECT}  —  Data Loading")
print("=" * 60)

df_raw = pd.read_csv(DATA_PATH)
print(f"Shape: {df_raw.shape}")

n_total  = len(df_raw)
n_events = df_raw[TARGET].sum()
print(f"Events: {n_events} / {n_total}  ({100*n_events/n_total:.1f}%)")
print(f"Imbalance ratio: {(n_total-n_events)/n_events:.1f}:1")

# ── 2. NLP Feature Extraction ─────────────────────────────────────────────────

print("\n[Step 2] NLP extraction from feature_0 …")

import re

MERCHANT_PATTERNS = {
    'Payroll':      [r'\bPAYROLL\b', r'\bDIRECT\s*DEP(OSIT)?\b',
                     r'\bDD\b.*\b(?:EMPLOYER|SALARY|WAGES)\b', r'\bACH\s*CR\b.*\bPAY\b'],
    'Transfer_ACH': [r'\bACH\s*(?:DEBIT|CREDIT|TRANSFER|PYMT|PMT)\b',
                     r'\bONLINE\s*TRANSFER\b', r'\bBANK\s*TRANSFER\b'],
    'Transfer_P2P': [r'\bVENMO\b', r'\bZELLE\b', r'\bCASH\s*APP\b',
                     r'\bSQUARE\s*CASH\b', r'\bPAYPAL\b'],
    'Subscription': [r'\bNETFLIX\b', r'\bSPOTIFY\b', r'\bHULU\b',
                     r'\bDISNEY\+?\b', r'\bAMAZON\s*PRIME\b',
                     r'\bAPPLE\s*(?:ONE|MUSIC|TV|ICLOUD)\b', r'\bSUBSCRIPTION\b'],
    'Grocery':      [r'\bWHOLE\s*FOODS\b', r'\bKROGER\b', r'\bSAFEWAY\b',
                     r'\bWALMART\b', r'\bTARGET\b', r'\bPUBLIX\b',
                     r'\bGROCERY\b', r'\bSUPERMARKET\b'],
    'Restaurant':   [r'\bMCDONALDS\b', r'\bSTARBUCKS\b', r'\bCHIPOTLE\b',
                     r'\bSUBWAY\b', r'\bPIZZA\b', r'\bRESTAURANT\b', r'\bDINING\b'],
    'FoodDelivery': [r'\bDOORDASH\b', r'\bUBER\s*EATS\b', r'\bGRUBHUB\b',
                     r'\bINSTACART\b', r'\bPOSTMATES\b'],
    'Rideshare':    [r'\bUBER\b(?!\s*EATS)', r'\bLYFT\b'],
    'Retail':       [r'\bAMAZON\b(?!\s*PRIME)', r'\bBEST\s*BUY\b',
                     r'\bHOME\s*DEPOT\b', r'\bETSY\b'],
    'Healthcare':   [r'\bPHARMACY\b', r'\bCVS\b', r'\bWALGREENS\b',
                     r'\bMEDICAL\b', r'\bHOSPITAL\b', r'\bDOCTOR\b'],
    'Entertainment':[r'\bAMC\b', r'\bTICKETMASTER\b', r'\bSTEAM\b',
                     r'\bGAMING\b', r'\bPLAYSTATION\b', r'\bXBOX\b'],
    'Travel':       [r'\bAIRLINE\b', r'\bFLIGHT\b', r'\bHOTEL\b', r'\bAIRBNB\b'],
    'Bank':         [r'\bATM\b', r'\bCASH\s*WITH?DRAW\b', r'\bBANK\s*FEE\b',
                     r'\bOVERDRAFT\b', r'\bCREDIT\s*CARD\s*PAYMENT\b'],
    'Transit':      [r'\bMTA\b', r'\bBART\b', r'\bMETRO\b',
                     r'\bBUS\s*(?:FARE|PASS)\b', r'\bTRANSIT\b'],
    'ATM':          [r'\bATM\s*(?:WITHDRAWAL|WD|CASH)\b'],
}

CHANNEL_PATTERNS = {
    'ACH':    [r'\bACH\b', r'\bAUTOMATED\s*CLEARING\b'],
    'Card':   [r'\bVISA\b', r'\bMASTERCARD\b', r'\bAMEX\b',
               r'\bDEBIT\b', r'\bCREDIT\b', r'\bPURCHASE\b', r'\bPOS\b'],
    'ATM':    [r'\bATM\b'],
    'Wire':   [r'\bWIRE\b', r'\bFED\s*WIRE\b'],
    'Mobile': [r'\bMOBILE\b', r'\bZELLE\b', r'\bVENMO\b', r'\bCASH\s*APP\b'],
    'Check':  [r'\bCHECK\b', r'\bCHEQUE\b'],
}

RISK_TIERS = {
    'High':   ['Transfer_P2P', 'ATM', 'Entertainment', 'Rideshare', 'Restaurant', 'FoodDelivery'],
    'Low':    ['Payroll', 'Healthcare', 'Grocery', 'Transit'],
    'Medium': ['Transfer_ACH', 'Subscription', 'Retail', 'Travel', 'Bank'],
}


def extract_nlp(text):
    if pd.isna(text):
        return {'merchant_category': 'Other', 'txn_channel': 'Other',
                'txn_direction': 'Unknown', 'is_recurring': 0,
                'is_p2p': 0, 'is_international': 0, 'merchant_risk_tier': 'Medium'}
    t = str(text).upper()
    mc = 'Other'
    for cat, pats in MERCHANT_PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            mc = cat; break
    ch = 'Other'
    for chan, pats in CHANNEL_PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            ch = chan; break
    direction = 'Unknown'
    if any(re.search(p, t) for p in [r'\bCREDIT\b', r'\bDEPOSIT\b', r'\bCR\b', r'\bREFUND\b']):
        direction = 'Credit'
    elif any(re.search(p, t) for p in [r'\bDEBIT\b', r'\bPURCHASE\b', r'\bPAYMENT\b',
                                        r'\bWITHDRAWAL\b', r'\bPAID\b']):
        direction = 'Debit'
    is_recurring     = 1 if re.search(r'\bRECURRING\b', t) else 0
    is_p2p           = 1 if any(re.search(p, t) for p in [r'\bVENMO\b', r'\bZELLE\b',
                                                            r'\bCASH\s*APP\b']) else 0
    is_international = 1 if any(re.search(p, t) for p in [r'\bINTL\b', r'\bINTERNATIONAL\b',
                                                            r'\bFOREIGN\b', r'\bFX\b']) else 0
    risk_tier = 'Medium'
    for tier, cats in RISK_TIERS.items():
        if mc in cats:
            risk_tier = tier; break
    return {'merchant_category': mc, 'txn_channel': ch, 'txn_direction': direction,
            'is_recurring': is_recurring, 'is_p2p': is_p2p,
            'is_international': is_international, 'merchant_risk_tier': risk_tier}


nlp_feats = df_raw['feature_0'].apply(extract_nlp)
df = pd.concat([df_raw, pd.DataFrame(list(nlp_feats), index=df_raw.index)], axis=1)

# Clean sentinels & encoding
df['feature_1'] = df['feature_1'].where(df['feature_1'] < 9999990, np.nan)
df['feature_8'] = pd.to_numeric(df['feature_8'].replace({'n': np.nan}), errors='coerce')

print(f"  NLP columns added: merchant_category, txn_channel, txn_direction, "
      f"is_recurring, is_p2p, is_international, merchant_risk_tier")

# ── 3. Train / Test Split ──────────────────────────────────────────────────────

print("\n[Step 3] Stratified 75/25 split …")

y = df[TARGET]
candidate_numeric = [f'feature_{i}' for i in range(1, 15)]
candidate_cat     = ['merchant_category', 'txn_channel', 'txn_direction', 'merchant_risk_tier']

X_raw_all = df[candidate_numeric + candidate_cat]
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw_all, y, test_size=0.25, stratify=y, random_state=SEED
)

print(f"  Train: {len(y_tr)}  events={y_tr.sum()} ({100*y_tr.mean():.1f}%)")
print(f"  Test : {len(y_te)}  events={y_te.sum()} ({100*y_te.mean():.1f}%)")

# ── 4. optbinning WoE / IV ────────────────────────────────────────────────────

print("\n[Step 4] optbinning WoE/IV with monotonic_trend='auto' …")


def fit_optbinning(feature_name, x_train, y_train, dtype='numerical'):
    """Fit OptimalBinning and return (ob, iv)."""
    x = x_train.values.astype(float) if dtype == 'numerical' else x_train.values.astype(str)
    valid = ~np.isnan(x.astype(float)) if dtype == 'numerical' else np.ones(len(x), bool)
    if valid.sum() < 10:
        return None, 0.0
    kwargs = dict(name=feature_name, dtype=dtype, solver='cp',
                  max_n_bins=10, min_bin_size=0.03)
    if dtype == 'numerical':
        kwargs['monotonic_trend'] = 'auto'
    ob = OptimalBinning(**kwargs)
    ob.fit(x[valid] if dtype == 'numerical' else x, y_train.values[valid] if dtype == 'numerical' else y_train.values)
    try:
        bt = ob.binning_table.build()
        iv = bt['IV'].iloc[:-2].sum()  # exclude totals rows
    except Exception:
        iv = 0.0
    return ob, float(iv)


ob_dict = {}
iv_rows = []
for col in candidate_numeric:
    x = X_tr_raw[col]
    ob, iv = fit_optbinning(col, x, y_tr, dtype='numerical')
    ob_dict[col] = ob
    iv_rows.append({'Feature': col, 'IV': round(iv, 4)})
    print(f"  {col:15s}  IV={iv:.4f}")

for col in candidate_cat:
    x = X_tr_raw[col].fillna('Missing')
    ob, iv = fit_optbinning(col, x, y_tr, dtype='categorical')
    ob_dict[col] = ob
    iv_rows.append({'Feature': col, 'IV': round(iv, 4)})
    print(f"  {col:25s}  IV={iv:.4f}")

iv_df = pd.DataFrame(iv_rows).sort_values('IV', ascending=False).reset_index(drop=True)


def iv_strength(iv):
    if iv >= 1.0:   return 'Suspected Leakage'
    if iv >= 0.5:   return 'Very Strong'
    if iv >= 0.3:   return 'Strong'
    if iv >= 0.1:   return 'Medium'
    if iv >= 0.02:  return 'Weak'
    return 'Useless'


iv_df['Strength'] = iv_df['IV'].apply(iv_strength)
print("\nIV Table:")
print(iv_df.to_string(index=False))
iv_df.to_csv(f'{DATA_DIR}/iv_table_optbinning.csv', index=False)

# Feature selection
selected_iv = iv_df[(iv_df['IV'] < 1.0) & (iv_df['IV'] >= 0.02)]['Feature'].tolist()
print(f"\nSelected by IV ({len(selected_iv)}): {selected_iv}")

# ── 5. WoE Encoding ───────────────────────────────────────────────────────────

print("\n[Step 5] Applying WoE encoding …")


def apply_woe(X_df, ob_dict, features):
    out = pd.DataFrame(index=X_df.index)
    for col in features:
        ob = ob_dict.get(col)
        if ob is None:
            out[col] = 0.0
            continue
        try:
            dtype = ob.dtype
            x = X_df[col].values.astype(float) if dtype == 'numerical' else X_df[col].fillna('Missing').values.astype(str)
            out[col] = ob.transform(x, metric='woe')
        except Exception as e:
            print(f"  WoE transform failed for {col}: {e}")
            out[col] = 0.0
    return out


X_tr_woe = apply_woe(X_tr_raw, ob_dict, selected_iv)
X_te_woe = apply_woe(X_te_raw, ob_dict, selected_iv)

# Fill any NaNs from transform
X_tr_woe = X_tr_woe.fillna(0.0)
X_te_woe = X_te_woe.fillna(0.0)

print(f"  WoE encoded: {list(X_tr_woe.columns)}")

# ── 5b. WoE PDP Plots ─────────────────────────────────────────────────────────

print("\n[Step 5b] WoE PDP plots (optbinning style) …")

num_feats_for_pdp = [f for f in selected_iv if f.startswith('feature_') and f in candidate_numeric][:6]
ncols = 3
nrows = (len(num_feats_for_pdp) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

for idx, feat in enumerate(num_feats_for_pdp):
    ax = axes[idx]
    ob = ob_dict.get(feat)
    if ob is None:
        ax.set_visible(False)
        continue
    try:
        bt = ob.binning_table.build()
        bins_plot = bt.iloc[:-2]  # drop totals
        x_pos = range(len(bins_plot))
        ax2 = ax.twinx()
        # Bar: count non-events / events
        ax.bar(x_pos, bins_plot['Count (%)'], color='#5b9bd5', alpha=0.55, label='% Count')
        ax2.plot(x_pos, bins_plot['Event rate'], color='#c00000', marker='o',
                 linewidth=2, markersize=5, label='Event rate')
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels([str(b)[:18] for b in bins_plot.index], rotation=35,
                           ha='right', fontsize=7)
        ax.set_ylabel('Bin %', fontsize=9)
        ax2.set_ylabel('Event rate', color='#c00000', fontsize=9)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        # Show auto-detected monotonic trend
        mt_label = getattr(ob, '_monotonic_trend_event_rate', 'auto')
        ax.text(0.98, 0.97, f'monotonic: {mt_label}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color='grey')
    except Exception as e:
        ax.set_title(f'{feat} (err)', fontsize=9)
        ax.text(0.5, 0.5, str(e)[:60], transform=ax.transAxes, ha='center', fontsize=7)

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('WoE PDP — optbinning with monotonic_trend="auto"', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_optbinning_woe_pdp.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  Saved: fig_optbinning_woe_pdp.png")

# ── 6. SHAP Recursive Feature Elimination ────────────────────────────────────

print("\n[Step 6] ShapRFECV …")

from probatus.feature_elimination import ShapRFECV

lgbm_rfe = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    num_leaves=20, min_child_samples=30, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
    random_state=SEED, n_jobs=-1, verbose=-1,
)

shap_rfe = ShapRFECV(
    lgbm_rfe, step=0.2, cv=5, scoring='roc_auc',
    n_jobs=1, random_state=SEED
)
results_df = shap_rfe.fit_compute(X_tr_woe, y_tr)
best_n = results_df.loc[results_df['val_metric_mean'].idxmax(), 'num_features']
selected_features = shap_rfe.get_reduced_features_set(num_features=int(best_n))
print(f"  Best n_features={best_n}  val_AUC={results_df['val_metric_mean'].max():.4f}")
print(f"  Selected: {selected_features}")

# Plot ShapRFECV elbow
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(results_df['num_features'], results_df['val_metric_mean'],
            yerr=results_df.get('val_metric_std', 0), fmt='o-', color='#1f4e79',
            capsize=3, linewidth=2, markersize=5)
ax.axvline(best_n, color='#c00000', linestyle='--', label=f'Selected: {best_n} features')
ax.set_xlabel('Number of Features', fontsize=11)
ax.set_ylabel('CV AUC (val)', fontsize=11)
ax.set_title('ShapRFECV — Feature Elimination Curve', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
fig.savefig(f'{FIG_DIR}/fig_shap_rfe_optbinning.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  Saved: fig_shap_rfe_optbinning.png")

X_tr_sel = X_tr_woe[selected_features]
X_te_sel = X_te_woe[selected_features]

# ── 7. BayesLGBM Training ─────────────────────────────────────────────────────

print("\n[Step 7] Bayesian hyperparameter optimisation …")

PBOUNDS = {
    'n_estimators':      (200, 600),
    'learning_rate':     (0.005, 0.05),
    'max_depth':         (3, 6),
    'num_leaves':        (10, 50),
    'min_child_samples': (30, 150),
    'reg_alpha':         (0.0, 2.0),
    'reg_lambda':        (0.0, 2.0),
    'colsample_bytree':  (0.4, 0.9),
    'subsample':         (0.6, 0.95),
}

GAP_THRESHOLD = 0.02
GAP_PENALTY   = 5.0

def objective_fn(n_estimators, learning_rate, max_depth, num_leaves,
                 min_child_samples, reg_alpha, reg_lambda, colsample_bytree, subsample):
    params = dict(
        n_estimators=int(n_estimators), learning_rate=learning_rate,
        max_depth=int(max_depth), num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        colsample_bytree=colsample_bytree, subsample=subsample,
        subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_aucs, val_aucs = [], []
    for tr_idx, val_idx in skf.split(X_tr_sel, y_tr):
        Xf = X_tr_sel.iloc[tr_idx]; Xv = X_tr_sel.iloc[val_idx]
        yf = y_tr.iloc[tr_idx];     yv = y_tr.iloc[val_idx]
        m  = lgb.LGBMClassifier(**params)
        m.fit(Xf, yf)
        tr_aucs.append(roc_auc_score(yf, m.predict_proba(Xf)[:, 1]))
        val_aucs.append(roc_auc_score(yv, m.predict_proba(Xv)[:, 1]))
    gap     = np.mean(tr_aucs) - np.mean(val_aucs)
    penalty = GAP_PENALTY * max(0, gap - GAP_THRESHOLD)
    return np.mean(val_aucs) - penalty


optimizer = BayesianOptimization(f=objective_fn, pbounds=PBOUNDS,
                                  random_state=SEED, verbose=0)
optimizer.maximize(init_points=10, n_iter=40)

raw = optimizer.max['params']
best_params = dict(
    n_estimators=int(raw['n_estimators']), learning_rate=raw['learning_rate'],
    max_depth=int(raw['max_depth']), num_leaves=int(raw['num_leaves']),
    min_child_samples=int(raw['min_child_samples']),
    reg_alpha=raw['reg_alpha'], reg_lambda=raw['reg_lambda'],
    colsample_bytree=raw['colsample_bytree'], subsample=raw['subsample'],
    subsample_freq=1,
)
print(f"  Best params: {best_params}")

final_model = lgb.LGBMClassifier(**best_params, random_state=SEED, n_jobs=-1, verbose=-1)
final_model.fit(X_tr_sel, y_tr)

tr_proba = final_model.predict_proba(X_tr_sel)[:, 1]
te_proba = final_model.predict_proba(X_te_sel)[:, 1]

tr_auc = roc_auc_score(y_tr, tr_proba)
te_auc = roc_auc_score(y_te, te_proba)
tr_ks  = ks_2samp(tr_proba[y_tr == 1], tr_proba[y_tr == 0]).statistic
te_ks  = ks_2samp(te_proba[y_te == 1], te_proba[y_te == 0]).statistic
gap    = tr_auc - te_auc

print(f"  Train AUC={tr_auc:.4f}  KS={tr_ks:.4f}")
print(f"  Test  AUC={te_auc:.4f}  KS={te_ks:.4f}")
print(f"  Gap   AUC={gap:.4f}  {'✓ within 2pp' if gap < 0.02 else '✗ EXCEEDS TARGET'}")

# ── 8. Evaluation Figures ─────────────────────────────────────────────────────

print("\n[Step 8] Generating evaluation figures …")

# ── 8a. ROC + KS (side by side, train vs test overlay) ────────────────────────

def _ks_curve(y_true, y_score):
    thresholds = np.sort(y_score)[::-1]
    tpr, fpr = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
    ks_val = max(np.array(tpr) - np.array(fpr))
    ks_thr = thresholds[np.argmax(np.array(tpr) - np.array(fpr))]
    return np.array(fpr), np.array(tpr), ks_val, ks_thr


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ROC
for (y_true, y_score, label, col, ls) in [
        (y_tr, tr_proba, f'Train (AUC={tr_auc:.3f})', '#1f4e79', '-'),
        (y_te, te_proba, f'Test  (AUC={te_auc:.3f})', '#c00000',  '--')]:
    fpr_r, tpr_r, _ = roc_curve(y_true, y_score)
    ax1.plot(fpr_r, tpr_r, color=col, linestyle=ls, linewidth=2, label=label)
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# KS
for (y_true, y_score, label, col, ls, ks_val) in [
        (y_tr, tr_proba, f'Train (KS={tr_ks:.3f})', '#1f4e79', '-', tr_ks),
        (y_te, te_proba, f'Test  (KS={te_ks:.3f})', '#c00000',  '--', te_ks)]:
    fpr_k, tpr_k, ks_v, ks_t = _ks_curve(y_true.values, y_score)
    ax2.plot(fpr_k, tpr_k, color=col, linestyle=ls, linewidth=2, label=f'TPR {label}')
    ax2.plot(fpr_k, fpr_k, color=col, linestyle=':', linewidth=1.5)

ax2.set_xlabel('Threshold (percentile)', fontsize=11)
ax2.set_ylabel('Cumulative Rate', fontsize=11)
ax2.set_title('KS Statistic', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

fig.suptitle(PROJECT, fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_roc_ks_optbinning.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  Saved: fig_roc_ks_optbinning.png")

# ── 8b. Confusion Matrix ──────────────────────────────────────────────────────

threshold = 0.5
y_pred_te = (te_proba >= threshold).astype(int)
cm = confusion_matrix(y_te, y_pred_te)
prec = precision_score(y_te, y_pred_te, zero_division=0)
rec  = recall_score(y_te, y_pred_te, zero_division=0)
f1   = f1_score(y_te, y_pred_te, zero_division=0)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Non-Default', 'Default'],
            yticklabels=['Non-Default', 'Default'])
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'Confusion Matrix  (threshold={threshold})\nPrec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}',
             fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_confusion_optbinning.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  Saved: fig_confusion_optbinning.png")

# ── 8c. SHAP Bar Importance ───────────────────────────────────────────────────

explainer  = shap.TreeExplainer(final_model)
shap_vals  = explainer.shap_values(X_te_sel)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

# Bar chart: mean |SHAP|
mean_shap = np.abs(shap_vals).mean(axis=0)
shap_df   = pd.DataFrame({'feature': selected_features, 'importance': mean_shap})
shap_df   = shap_df.sort_values('importance')

fig, ax = plt.subplots(figsize=(8, max(4, len(selected_features) * 0.5 + 1)))
colors = ['#1f4e79' if v >= mean_shap.mean() else '#5b9bd5' for v in shap_df['importance']]
ax.barh(shap_df['feature'], shap_df['importance'], color=colors, edgecolor='white')
ax.set_xlabel('Mean |SHAP value|', fontsize=11)
ax.set_title('SHAP Feature Importance (Test Set)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_shap_bar_optbinning.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  Saved: fig_shap_bar_optbinning.png")

# SHAP summary (beeswarm)
fig, ax = plt.subplots(figsize=(8, max(4, len(selected_features) * 0.5 + 1)))
shap.summary_plot(shap_vals, X_te_sel, feature_names=selected_features,
                  show=False, plot_size=None)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_shap_summary_optbinning.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  Saved: fig_shap_summary_optbinning.png")

# ── 8d. Expected vs Actual Decile Chart ──────────────────────────────────────

te_df = pd.DataFrame({'y': y_te.values, 'p': te_proba})
te_df['decile'] = pd.qcut(te_df['p'], 10, labels=False, duplicates='drop')
decile_tbl = te_df.groupby('decile').agg(
    n=('y', 'count'),
    events=('y', 'sum'),
    mean_pred=('p', 'mean')
).reset_index()
decile_tbl['actual_rate']   = decile_tbl['events'] / decile_tbl['n']
decile_tbl['expected_rate'] = decile_tbl['mean_pred']
decile_tbl['decile_label']  = (decile_tbl['decile'] + 1).astype(str)
decile_tbl.to_csv(f'{DATA_DIR}/table_decile_optbinning.csv', index=False)

x_pos = np.arange(len(decile_tbl))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x_pos - width/2, decile_tbl['expected_rate'] * 100,
       width, label='Expected (predicted)', color='#1f4e79', alpha=0.8)
ax.bar(x_pos + width/2, decile_tbl['actual_rate'] * 100,
       width, label='Actual (observed)',    color='#c00000', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'D{d}' for d in decile_tbl['decile_label']], fontsize=9)
ax.set_xlabel('Score Decile (low → high risk)', fontsize=11)
ax.set_ylabel('Default Rate (%)', fontsize=11)
ax.set_title('Expected vs Actual Default Rate by Score Decile', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_decile_optbinning.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  Saved: fig_decile_optbinning.png")

# ── 8e. Prediction Distribution ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(te_proba[y_te == 0], bins=40, density=True, alpha=0.6,
        color='#5b9bd5', label='Non-Default (y=0)')
ax.hist(te_proba[y_te == 1], bins=40, density=True, alpha=0.6,
        color='#c00000', label='Default (y=1)')
ax.set_xlabel('Predicted Probability of Default', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Score Distribution — Test Set', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_score_dist_optbinning.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  Saved: fig_score_dist_optbinning.png")

# ── 8f. Correlation Heatmap ───────────────────────────────────────────────────

corr = X_tr_sel.corr()
fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=mask, ax=ax, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix (WoE-encoded, Train)', fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIG_DIR}/fig_corr_optbinning.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  Saved: fig_corr_optbinning.png")

# ── 9. Persist ────────────────────────────────────────────────────────────────

print("\n[Step 9] Saving outputs …")

bundle = {
    'model':            final_model,
    'best_params':      best_params,
    'features':         selected_features,
    'ob_dict':          ob_dict,
    'X_tr':             X_tr_sel, 'X_te': X_te_sel,
    'y_tr':             y_tr,     'y_te': y_te,
    'tr_auc':           tr_auc,   'te_auc': te_auc,
    'tr_ks':            tr_ks,    'te_ks':  te_ks,
    'tr_proba':         tr_proba, 'te_proba': te_proba,
}
with open(f'{BASE}/final_model_optbinning.pkl', 'wb') as f:
    pickle.dump(bundle, f)
print(f"  Saved: final_model_optbinning.pkl")

# ── 10. Final Summary ─────────────────────────────────────────────────────────

top_decile = decile_tbl[decile_tbl['decile'] == decile_tbl['decile'].max()]['events'].sum()
top_decile_pct = 100 * top_decile / y_te.sum()  # % of test-set events

print()
print("=" * 55)
print(f"  {PROJECT}")
print(f"  Author: {AUTHOR}")
print("=" * 55)
print(f"  Dataset   : {n_total} rows | {int(n_events)} events ({100*n_events/n_total:.1f}%) | "
      f"ratio {(n_total-n_events)/n_events:.1f}:1")
print(f"  Features  : {selected_features}")
print(f"  {'':15s}   AUC      KS")
print(f"  Train     : {tr_auc:.4f}   {tr_ks:.4f}")
print(f"  Test      : {te_auc:.4f}   {te_ks:.4f}")
print(f"  Gap       : {gap:.4f}   {'✓ within 2pp' if gap < 0.02 else '✗ EXCEEDS TARGET'}")
print(f"  Top decile captures {top_decile_pct:.1f}% of events")
print("=" * 55)
print("\nFigures saved:")
for f in sorted(os.listdir(FIG_DIR)):
    if 'optbinning' in f:
        print(f"  {FIG_DIR}/{f}")
