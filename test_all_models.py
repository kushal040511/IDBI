"""
IDBI MSME Risk Intelligence -- Comprehensive Model Evaluation Suite
====================================================================
Trains models from scratch on project-matched synthetic data, then
evaluates ALL models with the full set of requested metrics:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * Log Loss
  * Area Under Curve (ROC-AUC)
  * Confusion Matrix

Models evaluated:
  A. XGBoost MSME Engine      -- 16 MSME behavioral features
  B. Logistic Regression      -- 3-feature baseline
  C. XGBoost Real Credit      -- sklearn Pipeline (num + cat features)

Run:
    python test_all_models.py
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Force UTF-8 on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, confusion_matrix,
    roc_curve, average_precision_score, precision_recall_curve,
)
import xgboost as xgb

BASE       = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE, 'evaluation_report')
os.makedirs(REPORT_DIR, exist_ok=True)

SEED = 42

PALETTE = {
    'xgb_msme': '#6C63FF',
    'lr_base':  '#FF6584',
    'xgb_real': '#43E97B',
    'bg':       '#0F0F1A',
    'surface':  '#1A1A2E',
    'text':     '#E8E8F0',
    'grid':     '#2A2A3E',
}

plt.rcParams.update({
    'figure.facecolor': PALETTE['bg'],
    'axes.facecolor':   PALETTE['surface'],
    'axes.edgecolor':   PALETTE['grid'],
    'axes.labelcolor':  PALETTE['text'],
    'xtick.color':      PALETTE['text'],
    'ytick.color':      PALETTE['text'],
    'text.color':       PALETTE['text'],
    'grid.color':       PALETTE['grid'],
    'grid.linewidth':   0.5,
    'font.family':      'sans-serif',
    'font.size':        10,
})

# ==================================================================
#  HELPERS
# ==================================================================
def banner(title):
    w = 68
    print("\n" + "=" * w)
    print("  " + title)
    print("=" * w)

def sub(title):
    pad = max(2, 60 - len(title))
    print("\n-- " + title + " " + "-" * pad)

def full_report(y_true, probs, threshold, label):
    """Print + return all 7 requested metrics."""
    y_pred = (probs >= threshold).astype(int)
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    ll     = log_loss(y_true, probs)
    roc    = roc_auc_score(y_true, probs)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  Threshold   : {threshold:.2f}")
    print(f"  Accuracy    : {acc*100:6.2f}%")
    print(f"  Precision   : {prec*100:6.2f}%")
    print(f"  Recall      : {rec*100:6.2f}%    <- defaults caught")
    print(f"  F1 Score    : {f1*100:6.2f}%")
    print(f"  Log Loss    : {ll:.4f}")
    print(f"  ROC-AUC     : {roc:.4f}")
    print()
    print(f"  Confusion Matrix:")
    print(f"                    Pred Healthy   Pred Default")
    print(f"    Act Healthy  |   {tn:8,}     {fp:8,}   (FP)")
    print(f"    Act Default  |   {fn:8,}     {tp:8,}   (TP)")
    print(f"    Specificity  : {tn/(tn+fp)*100:.2f}%   (true negative rate)")
    print(f"    Miss rate    : {fn/(fn+tp)*100:.2f}%   (false negative rate)")

    return dict(
        label=label, threshold=threshold,
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        log_loss=ll, roc_auc=roc,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        probs=probs, y_true=y_true,
    )

# ==================================================================
#  DATA GENERATORS
# ==================================================================
def make_msme_data(n=30_000, seed=SEED):
    """
    Synthetic MSME borrower dataset.
    Mirrors the feature engineering in train_real_model.py.
    Default rate ~17%.
    """
    rng = np.random.default_rng(seed)
    d   = (rng.random(n) < 0.17).astype(int)          # default flag

    df = pd.DataFrame()
    df['default_flag']           = d
    df['revolving_utilization']  = np.clip(rng.beta(2, 5, n) + d * rng.uniform(0.1, 0.4, n), 0, 1)
    df['debt_ratio']             = np.clip(rng.exponential(0.35, n) + d * rng.uniform(0.1, 0.5, n), 0, 5)
    df['late_30_59']             = (rng.poisson(0.3, n) + d * rng.poisson(1.5, n)).astype(int)
    df['late_60_89']             = (rng.poisson(0.1, n) + d * rng.poisson(0.8, n)).astype(int)
    df['late_90_days']           = (rng.poisson(0.05, n) + d * rng.poisson(0.5, n)).astype(int)
    df['open_credit_lines']      = rng.integers(2, 20, n)
    df['real_estate_loans']      = rng.integers(0, 4, n)
    df['num_dependents']         = rng.integers(0, 5, n)
    df['income_stability']       = np.clip(rng.beta(5, 2, n) - d * 0.2, 0, 1)
    df['gst_compliance_score']   = np.clip(0.85 - d * 0.25 + rng.normal(0, 0.05, n), 0, 1)
    df['emi_delay_count']        = np.clip(df['late_30_59'] + df['late_60_89'], 0, 12)
    df['cashflow_stress_ratio']  = np.clip(df['debt_ratio'] * rng.uniform(0.9, 1.1, n), 0, 5)
    df['working_capital_usage']  = np.clip(df['revolving_utilization'] * 0.6 + rng.normal(0.1, 0.05, n), 0, 1)
    df['revenue_trend_index']    = np.clip(1.2 - df['debt_ratio'] * 0.4 + rng.normal(0, 0.1, n), 0.2, 2.0)
    df['payment_history_score']  = np.clip(1 - df['late_90_days'] * 0.2 - df['late_30_59'] * 0.05, 0, 1)
    df['supplier_payment_risk']  = ((df['late_30_59'] > 2).astype(float)
                                    + (df['late_90_days'] > 0).astype(float))
    return df


def make_credit_data(n=32_000, seed=SEED):
    """
    Synthetic credit_risk_dataset-style borrowers.
    Default rate ~22%.  Mirrors train_real_credit.py feature space.
    """
    rng = np.random.default_rng(seed)
    y   = (rng.random(n) < 0.22).astype(int)

    grade_map  = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    grade_keys = list(grade_map.keys())
    home_opts  = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    intent_opts = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE',
                   'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']

    df = pd.DataFrame()
    df['loan_status']                = y
    df['person_age']                 = np.clip(rng.normal(28, 7, n).astype(int), 18, 80)
    df['person_income']              = np.clip(
        rng.lognormal(10.7, 0.5, n) - y * rng.uniform(5000, 15000, n), 4000, 6_000_000)
    df['person_emp_length']          = np.clip(rng.exponential(4, n), 0, 41)
    df['loan_amnt']                  = np.clip(rng.lognormal(8.8, 0.6, n), 500, 35_000)
    df['loan_int_rate']              = np.clip(
        rng.normal(11, 5, n) + y * rng.uniform(0, 5, n), 5.42, 23.22)
    df['loan_percent_income']        = np.clip(df['loan_amnt'] / df['person_income'], 0.0, 0.66)
    df['cb_person_cred_hist_length'] = np.clip(rng.normal(5.8, 3.5, n).astype(int), 2, 30)
    df['loan_grade']                 = rng.choice(grade_keys, n,
                                                   p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.07, 0.03])
    df['loan_grade_ord']             = df['loan_grade'].map(grade_map)
    df['default_flag']               = (rng.random(n) < 0.18).astype(int)
    df['note_stress_index']          = np.clip(
        y * rng.uniform(0.4, 0.8, n) + rng.normal(0.1, 0.05, n), 0, 1)
    df['person_home_ownership']      = rng.choice(home_opts, n, p=[0.46, 0.38, 0.10, 0.06])
    df['loan_intent']                = rng.choice(intent_opts, n,
                                                   p=[0.20, 0.20, 0.18, 0.14, 0.14, 0.14])
    return df


# ==================================================================
#  SECTION 1 — GENERATE DATA & SPLIT
# ==================================================================
banner("STEP 1/4  --  GENERATING SYNTHETIC DATASETS")

msme_df  = make_msme_data(30_000)
cred_df  = make_credit_data(32_000)

MSME_FEATURES = [
    'revolving_utilization', 'debt_ratio', 'late_30_59', 'late_60_89', 'late_90_days',
    'open_credit_lines', 'real_estate_loans', 'num_dependents', 'income_stability',
    'gst_compliance_score', 'emi_delay_count', 'cashflow_stress_ratio',
    'working_capital_usage', 'revenue_trend_index', 'payment_history_score',
    'supplier_payment_risk',
]
LR_FEATURES = ['revolving_utilization', 'debt_ratio', 'late_90_days']

NUM_COLS = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'loan_grade_ord', 'default_flag', 'note_stress_index']
CAT_COLS = ['person_home_ownership', 'loan_intent']

X_msme = msme_df[MSME_FEATURES].fillna(msme_df[MSME_FEATURES].median())
y_msme = msme_df['default_flag'].values

X_cred = cred_df[NUM_COLS + CAT_COLS]
y_cred = cred_df['loan_status'].values

Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(X_msme, y_msme, test_size=0.25,
                                                random_state=SEED, stratify=y_msme)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cred, y_cred, test_size=0.20,
                                                random_state=SEED, stratify=y_cred)

print(f"  MSME dataset     : {len(X_msme):,} samples | default rate {y_msme.mean()*100:.1f}%")
print(f"    Train {len(Xm_tr):,}  |  Test {len(Xm_te):,}")
print(f"  Credit dataset   : {len(X_cred):,} samples | default rate {y_cred.mean()*100:.1f}%")
print(f"    Train {len(Xc_tr):,}  |  Test {len(Xc_te):,}")

# ==================================================================
#  SECTION 2 — TRAIN MODELS
# ==================================================================
banner("STEP 2/4  --  TRAINING MODELS")

# ---- Model A: XGBoost MSME (16 features) -------------------------
print("\n  [A] XGBoost MSME Engine  (16 MSME behavioral features)")
imputer_msme = SimpleImputer(strategy='median')
Xm_tr_imp = imputer_msme.fit_transform(Xm_tr)
Xm_te_imp = imputer_msme.transform(Xm_te)

spw_msme = float((ym_tr == 0).sum() / (ym_tr == 1).sum())
xgb_msme = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.85,
    reg_lambda=2, reg_alpha=0.5,
    scale_pos_weight=spw_msme,
    eval_metric='auc', random_state=SEED, n_jobs=-1,
)
xgb_msme.fit(Xm_tr_imp, ym_tr,
             eval_set=[(Xm_te_imp, ym_te)], verbose=False)
print(f"     XGBoost MSME trained  ({xgb_msme.n_estimators} trees)")

# ---- Model B: Logistic Regression baseline (3 features) ----------
print("  [B] Logistic Regression Baseline  (3 features)")
lr_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler()),
    ('lr',  LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED)),
])
lr_pipe.fit(Xm_tr[LR_FEATURES], ym_tr)
print("     Logistic Regression trained")

# ---- Model C: XGBoost Real Credit Pipeline -----------------------
print("  [C] XGBoost Real Credit  (full sklearn Pipeline)")

pre = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), NUM_COLS),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
])
spw_cred = float((yc_tr == 0).sum() / (yc_tr == 1).sum())
real_pipe = Pipeline([
    ('pre', pre),
    ('clf', xgb.XGBClassifier(
        n_estimators=550, max_depth=6, learning_rate=0.057,
        subsample=0.878, colsample_bytree=0.848,
        min_child_weight=10, gamma=3.2,
        reg_lambda=0.069, reg_alpha=0.021,
        scale_pos_weight=spw_cred,
        eval_metric='auc', random_state=SEED, n_jobs=-1,
    )),
])
real_pipe.fit(Xc_tr, yc_tr)
print("     XGBoost Real Credit Pipeline trained")

# ==================================================================
#  SECTION 3 — PREDICT
# ==================================================================
banner("STEP 3/4  --  SCORING TEST SETS")

probs_xgb_msme = xgb_msme.predict_proba(Xm_te_imp)[:, 1]
probs_lr       = lr_pipe.predict_proba(Xm_te[LR_FEATURES])[:, 1]
probs_real     = real_pipe.predict_proba(Xc_te)[:, 1]

print(f"  Scored {len(ym_te):,} MSME test samples")
print(f"  Scored {len(yc_te):,} Credit test samples")

# ==================================================================
#  SECTION 4 — FULL METRIC REPORTS
# ==================================================================
banner("STEP 4/4  --  FULL METRIC REPORTS")

all_results = {}

# ---- MODEL A: XGBoost MSME @0.50 --------------------------------
sub("MODEL A -- XGBoost MSME (16 features)  @  threshold = 0.50")
r_a50 = full_report(ym_te, probs_xgb_msme, 0.50, "XGBoost MSME @0.50")
all_results["XGBoost MSME @0.50"] = r_a50

sub("MODEL A -- XGBoost MSME (16 features)  @  threshold = 0.30  (early-warning)")
r_a30 = full_report(ym_te, probs_xgb_msme, 0.30, "XGBoost MSME @0.30")
all_results["XGBoost MSME @0.30"] = r_a30

# ---- MODEL B: Logistic Regression @0.50 -------------------------
sub("MODEL B -- Logistic Regression Baseline (3 features)  @  threshold = 0.50")
r_b50 = full_report(ym_te, probs_lr, 0.50, "LR Baseline @0.50")
all_results["LR Baseline @0.50"] = r_b50

# ---- MODEL C: XGBoost Real Credit --------------------------------
sub("MODEL C -- XGBoost Real Credit Pipeline  @  threshold = 0.50")
r_c50 = full_report(yc_te, probs_real, 0.50, "XGBoost Real @0.50")
all_results["XGBoost Real @0.50"] = r_c50

sub("MODEL C -- XGBoost Real Credit Pipeline  @  threshold = 0.63  (recall>=85%)")
r_c63 = full_report(yc_te, probs_real, 0.63, "XGBoost Real @0.63")
all_results["XGBoost Real @0.63"] = r_c63

# ---- Summary table -----------------------------------------------
banner("SUMMARY COMPARISON TABLE -- ALL MODELS x ALL METRICS")
hdr = f"  {'Model':<26}  {'Acc%':>6}  {'Prec%':>6}  {'Rec%':>6}  {'F1%':>6}  {'LogLoss':>8}  {'ROC-AUC':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for name, r in all_results.items():
    print(f"  {name:<26}  {r['accuracy']*100:6.2f}  {r['precision']*100:6.2f}  "
          f"{r['recall']*100:6.2f}  {r['f1']*100:6.2f}  {r['log_loss']:8.4f}  {r['roc_auc']:8.4f}")

# ==================================================================
#  VISUALIZATIONS
# ==================================================================
banner("GENERATING EVALUATION CHARTS")

colors_map = {
    "XGBoost MSME @0.50": PALETTE['xgb_msme'],
    "XGBoost MSME @0.30": '#A09FFF',
    "LR Baseline @0.50":  PALETTE['lr_base'],
    "XGBoost Real @0.50": PALETTE['xgb_real'],
    "XGBoost Real @0.63": '#7BFFB8',
}

# -- Fig 1: ROC Curves --------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 6))
fig1.patch.set_facecolor(PALETTE['bg'])
roc_triples = [
    (ym_te, probs_xgb_msme, PALETTE['xgb_msme'], 'XGBoost MSME (16 features)'),
    (ym_te, probs_lr,        PALETTE['lr_base'],  'LR Baseline (3 features)'),
    (yc_te, probs_real,      PALETTE['xgb_real'], 'XGBoost Real Credit Pipeline'),
]
for y_t, p, col, lbl in roc_triples:
    fpr, tpr, _ = roc_curve(y_t, p)
    auc_v = roc_auc_score(y_t, p)
    ax1.plot(fpr, tpr, color=col, lw=2.5, label=f"{lbl}  (AUC={auc_v:.4f})")
ax1.plot([0, 1], [0, 1], 'w--', alpha=0.3, lw=1, label='Random Classifier')
ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.01])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves -- All Models', fontsize=14, fontweight='bold', pad=14)
ax1.legend(loc='lower right', framealpha=0.2, fontsize=9)
ax1.grid(True, alpha=0.4)
fig1.tight_layout()
fig1.savefig(os.path.join(REPORT_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  [OK] roc_curves.png")

# -- Fig 2: Precision-Recall Curves --------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 6))
fig2.patch.set_facecolor(PALETTE['bg'])
for y_t, p, col, lbl in roc_triples:
    pr, rc, _ = precision_recall_curve(y_t, p)
    ap = average_precision_score(y_t, p)
    ax2.plot(rc, pr, color=col, lw=2.5, label=f"{lbl}  (AP={ap:.4f})")
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.01])
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curves -- All Models', fontsize=14, fontweight='bold', pad=14)
ax2.legend(loc='upper right', framealpha=0.2, fontsize=9)
ax2.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig(os.path.join(REPORT_DIR, 'pr_curves.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  [OK] pr_curves.png")

# -- Fig 3: Confusion Matrices (5 panels) --------------------------
cm_panels = [
    (r_a50, 'XGBoost MSME\n@ thr=0.50'),
    (r_a30, 'XGBoost MSME\n@ thr=0.30'),
    (r_b50, 'LR Baseline\n@ thr=0.50'),
    (r_c50, 'XGBoost Real\n@ thr=0.50'),
    (r_c63, 'XGBoost Real\n@ thr=0.63'),
]
fig3, axes3 = plt.subplots(1, 5, figsize=(22, 5))
fig3.patch.set_facecolor(PALETTE['bg'])
fig3.suptitle('Confusion Matrices -- All Models & Thresholds',
              fontsize=14, fontweight='bold', color=PALETTE['text'], y=1.03)
for ax, (r, title) in zip(axes3, cm_panels):
    cm_arr = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
    im = ax.imshow(cm_arr, cmap='RdPu', aspect='auto')
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            ax.text(j, i, f"{labels[i][j]}\n{val:,}",
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white' if val < cm_arr.max() * 0.6 else 'black')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Healthy', 'Pred Default'], fontsize=8)
    ax.set_yticklabels(['Act Healthy', 'Act Default'], fontsize=8)
    ax.set_title(f"{title}\nAcc {r['accuracy']*100:.1f}%  F1 {r['f1']*100:.1f}%",
                 fontsize=9, color=PALETTE['text'])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig3.tight_layout()
fig3.savefig(os.path.join(REPORT_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  [OK] confusion_matrices.png")

# -- Fig 4: Grouped Metrics Bar Chart ------------------------------
metrics_keys   = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels  = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
model_names    = list(all_results.keys())
bar_colors     = [colors_map[n] for n in model_names]

x     = np.arange(len(metrics_keys))
bar_w = 0.15
fig4, ax4 = plt.subplots(figsize=(14, 6))
fig4.patch.set_facecolor(PALETTE['bg'])
for idx, (name, r) in enumerate(all_results.items()):
    vals = [r[m] for m in metrics_keys]
    bars = ax4.bar(x + idx * bar_w, vals, width=bar_w,
                   label=name, color=bar_colors[idx],
                   alpha=0.88, edgecolor='white', linewidth=0.3)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f'{h:.2f}', ha='center', va='bottom', fontsize=6,
                 color=PALETTE['text'])
ax4.set_xticks(x + bar_w * (len(model_names) - 1) / 2)
ax4.set_xticklabels(metric_labels, fontsize=11)
ax4.set_ylim(0, 1.15)
ax4.set_ylabel('Score (0 to 1)', fontsize=11)
ax4.set_title('All Metrics Comparison -- All Models', fontsize=14, fontweight='bold', pad=14)
ax4.legend(loc='upper left', framealpha=0.2, fontsize=8, ncol=2)
ax4.grid(True, axis='y', alpha=0.35)
fig4.tight_layout()
fig4.savefig(os.path.join(REPORT_DIR, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)
print("  [OK] metrics_comparison.png")

# -- Fig 5: Score Distributions -----------------------------------
fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5))
fig5.patch.set_facecolor(PALETTE['bg'])
fig5.suptitle('Predicted Probability Distributions  (Healthy vs Default)',
              fontsize=13, fontweight='bold', color=PALETTE['text'])
dist_data = [
    (ym_te, probs_xgb_msme, 'XGBoost MSME'),
    (ym_te, probs_lr,        'LR Baseline'),
    (yc_te, probs_real,      'XGBoost Real Credit'),
]
for ax, (y_t, p, title) in zip(axes5, dist_data):
    ax.hist(p[y_t == 0], bins=50, alpha=0.55, color='#60CFFF', label='Healthy', density=True)
    ax.hist(p[y_t == 1], bins=50, alpha=0.65, color='#FF6060', label='Default', density=True)
    ax.set_title(title, fontsize=11, color=PALETTE['text'])
    ax.set_xlabel('Predicted Probability', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=9, framealpha=0.2)
    ax.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(REPORT_DIR, 'score_distributions.png'), dpi=150, bbox_inches='tight')
plt.close(fig5)
print("  [OK] score_distributions.png")

# -- Fig 6: Log Loss comparison -----------------------------------
fig6, ax6 = plt.subplots(figsize=(9, 5))
fig6.patch.set_facecolor(PALETTE['bg'])
ll_names  = model_names
ll_values = [r['log_loss'] for r in all_results.values()]
bars6 = ax6.barh(ll_names, ll_values,
                  color=[colors_map[n] for n in ll_names],
                  alpha=0.85, edgecolor='white', linewidth=0.4)
for bar, val in zip(bars6, ll_values):
    ax6.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
             f'{val:.4f}', va='center', fontsize=9.5, color=PALETTE['text'])
ax6.set_xlabel('Log Loss  (lower is better)', fontsize=11)
ax6.set_title('Log Loss -- All Models', fontsize=13, fontweight='bold', pad=12)
ax6.grid(True, axis='x', alpha=0.3)
ax6.set_xlim(0, max(ll_values) * 1.3)
fig6.tight_layout()
fig6.savefig(os.path.join(REPORT_DIR, 'log_loss_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig6)
print("  [OK] log_loss_comparison.png")

# -- Fig 7: Big combined dashboard --------------------------------
fig7 = plt.figure(figsize=(20, 12))
fig7.patch.set_facecolor(PALETTE['bg'])
gs = GridSpec(2, 3, figure=fig7, hspace=0.45, wspace=0.35)

# ROC (top-left)
ax_roc = fig7.add_subplot(gs[0, 0])
for y_t, p, col, lbl in roc_triples:
    fpr, tpr, _ = roc_curve(y_t, p)
    auc_v = roc_auc_score(y_t, p)
    ax_roc.plot(fpr, tpr, color=col, lw=2, label=f"{lbl[:18]}\n(AUC={auc_v:.3f})")
ax_roc.plot([0, 1], [0, 1], 'w--', alpha=0.25, lw=1)
ax_roc.set_title('ROC Curves', fontweight='bold')
ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
ax_roc.legend(fontsize=7, framealpha=0.2)
ax_roc.grid(alpha=0.3)

# PR (top-centre)
ax_pr = fig7.add_subplot(gs[0, 1])
for y_t, p, col, lbl in roc_triples:
    pr_v, rc_v, _ = precision_recall_curve(y_t, p)
    ax_pr.plot(rc_v, pr_v, color=col, lw=2, label=lbl[:18])
ax_pr.set_title('Precision-Recall Curves', fontweight='bold')
ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
ax_pr.legend(fontsize=7, framealpha=0.2)
ax_pr.grid(alpha=0.3)

# Metrics bars (top-right)
ax_mb = fig7.add_subplot(gs[0, 2])
x2 = np.arange(len(metrics_keys))
bw2 = 0.16
for idx, (name, r) in enumerate(all_results.items()):
    vals = [r[m] for m in metrics_keys]
    ax_mb.bar(x2 + idx * bw2, vals, width=bw2, label=name,
              color=bar_colors[idx], alpha=0.85)
ax_mb.set_xticks(x2 + bw2 * (len(model_names)-1)/2)
ax_mb.set_xticklabels(metric_labels, fontsize=7, rotation=15)
ax_mb.set_ylim(0, 1.1); ax_mb.set_ylabel('Score')
ax_mb.set_title('All Metrics', fontweight='bold')
ax_mb.legend(fontsize=6, framealpha=0.2, ncol=2)
ax_mb.grid(axis='y', alpha=0.3)

# Score distributions (bottom row)
for col_idx, (y_t, p, title) in enumerate(dist_data):
    ax_d = fig7.add_subplot(gs[1, col_idx])
    ax_d.hist(p[y_t == 0], bins=40, alpha=0.5, color='#60CFFF', label='Healthy', density=True)
    ax_d.hist(p[y_t == 1], bins=40, alpha=0.65, color='#FF6060', label='Default', density=True)
    ax_d.set_title(f'Score Dist. -- {title}', fontweight='bold', fontsize=9)
    ax_d.set_xlabel('Predicted Probability', fontsize=8)
    ax_d.set_ylabel('Density', fontsize=8)
    ax_d.legend(fontsize=8, framealpha=0.2)
    ax_d.grid(alpha=0.3)

fig7.suptitle('IDBI MSME Risk Engine -- Full Model Evaluation Dashboard',
              fontsize=16, fontweight='bold', color=PALETTE['text'], y=1.01)
fig7.savefig(os.path.join(REPORT_DIR, 'dashboard.png'), dpi=150, bbox_inches='tight')
plt.close(fig7)
print("  [OK] dashboard.png")

# ==================================================================
#  SAVE JSON REPORT
# ==================================================================
def serialise(r):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else
                int(v)   if isinstance(v, (np.integer, int)) else
                str(v)   if not isinstance(v, (str, dict, list)) else v)
            for k, v in r.items() if k not in ('probs', 'y_true')}

report_path = os.path.join(REPORT_DIR, 'evaluation_results.json')
json_report = {name: serialise(r) for name, r in all_results.items()}
with open(report_path, 'w') as f:
    json.dump(json_report, f, indent=2)
print(f"  [OK] evaluation_results.json -> {report_path}")

# ==================================================================
#  FINAL SUMMARY
# ==================================================================
banner("EVALUATION COMPLETE -- FINAL HIGHLIGHTS")
print()
print(f"  {'Model':<26}  {'Acc%':>6}  {'F1%':>6}  {'Recall%':>8}  {'LogLoss':>9}  {'ROC-AUC':>9}")
print("  " + "-" * 72)
for name, r in all_results.items():
    print(f"  {name:<26}  {r['accuracy']*100:6.2f}  {r['f1']*100:6.2f}"
          f"  {r['recall']*100:8.2f}  {r['log_loss']:9.4f}  {r['roc_auc']:9.4f}")
print()
print("  Charts saved to:", REPORT_DIR)
print("    roc_curves.png | pr_curves.png | confusion_matrices.png")
print("    metrics_comparison.png | score_distributions.png")
print("    log_loss_comparison.png | dashboard.png")
print("    evaluation_results.json")
print()
banner("DONE")
