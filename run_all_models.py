"""
IDBI MSME Risk Engine -- Run ALL Models
=========================================
Trains and evaluates every model in the project pipeline:

  Model 1  -- Logistic Regression Baseline    (3 features)
  Model 2  -- XGBoost MSME Engine             (16 features)
  Model 3  -- XGBoost + NLP Stress Index      (17 features = 16 structured + NLP)
  Model 4  -- LightGBM MSME Engine            (16 features)
  Model 5  -- Optuna-Tuned XGBoost            (16+NLP, hyperparameter searched)
  Model 6  -- XGBoost Real Credit Pipeline    (full sklearn Pipeline, num+cat)

All models report:
  Accuracy / Precision / Recall / F1 / Log Loss / ROC-AUC / Confusion Matrix

Run:
    python run_all_models.py
"""

import sys, os, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, confusion_matrix,
    roc_curve, average_precision_score, precision_recall_curve,
)
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE       = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE, 'evaluation_report')
os.makedirs(REPORT_DIR, exist_ok=True)
SEED = 42

# ── colour palette ────────────────────────────────────────────────
COLORS = ['#6C63FF','#FF6584','#43E97B','#F7971E','#A18CD1','#FD7272']
BG, SURF, TXT, GRID = '#0F0F1A','#1A1A2E','#E8E8F0','#2A2A3E'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': SURF, 'axes.edgecolor': GRID,
    'axes.labelcolor': TXT, 'xtick.color': TXT, 'ytick.color': TXT,
    'text.color': TXT, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'font.family': 'sans-serif', 'font.size': 10,
})

# ==================================================================
#  HELPERS
# ==================================================================
def banner(title):
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70)

def section(title):
    print(f"\n  >> {title}")
    print("  " + "-" * 60)

def report_metrics(y_true, probs, threshold, name):
    y_pred = (probs >= threshold).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    ll   = log_loss(y_true, probs)
    roc  = roc_auc_score(y_true, probs)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  Accuracy    : {acc*100:6.2f}%")
    print(f"  Precision   : {prec*100:6.2f}%")
    print(f"  Recall      : {rec*100:6.2f}%    <- defaults caught")
    print(f"  F1 Score    : {f1*100:6.2f}%")
    print(f"  Log Loss    : {ll:.4f}")
    print(f"  ROC-AUC     : {roc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"                    Pred Healthy   Pred Default")
    print(f"    Act Healthy  |   {tn:8,}     {fp:8,}  (FP)")
    print(f"    Act Default  |   {fn:8,}     {tp:8,}  (TP)")
    return dict(name=name, threshold=threshold,
                accuracy=acc, precision=prec, recall=rec, f1=f1,
                log_loss=ll, roc_auc=roc,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                probs=probs, y_true=y_true)

# ==================================================================
#  DATA GENERATORS
# ==================================================================
def make_msme(n=30_000, seed=SEED):
    rng = np.random.default_rng(seed)
    d   = (rng.random(n) < 0.17).astype(int)
    df  = pd.DataFrame()
    df['default_flag']           = d
    df['revolving_utilization']  = np.clip(rng.beta(2,5,n) + d*rng.uniform(0.1,0.4,n), 0, 1)
    df['debt_ratio']             = np.clip(rng.exponential(0.35,n) + d*rng.uniform(0.1,0.5,n), 0, 5)
    df['late_30_59']             = (rng.poisson(0.3,n) + d*rng.poisson(1.5,n)).astype(int)
    df['late_60_89']             = (rng.poisson(0.1,n) + d*rng.poisson(0.8,n)).astype(int)
    df['late_90_days']           = (rng.poisson(0.05,n)+ d*rng.poisson(0.5,n)).astype(int)
    df['open_credit_lines']      = rng.integers(2, 20, n)
    df['real_estate_loans']      = rng.integers(0, 4, n)
    df['num_dependents']         = rng.integers(0, 5, n)
    df['income_stability']       = np.clip(rng.beta(5,2,n) - d*0.2, 0, 1)
    df['gst_compliance_score']   = np.clip(0.85 - d*0.25 + rng.normal(0,0.05,n), 0, 1)
    df['emi_delay_count']        = np.clip(df['late_30_59'] + df['late_60_89'], 0, 12)
    df['cashflow_stress_ratio']  = np.clip(df['debt_ratio']*rng.uniform(0.9,1.1,n), 0, 5)
    df['working_capital_usage']  = np.clip(df['revolving_utilization']*0.6 + rng.normal(0.1,0.05,n), 0, 1)
    df['revenue_trend_index']    = np.clip(1.2 - df['debt_ratio']*0.4 + rng.normal(0,0.1,n), 0.2, 2.0)
    df['payment_history_score']  = np.clip(1 - df['late_90_days']*0.2 - df['late_30_59']*0.05, 0, 1)
    df['supplier_payment_risk']  = ((df['late_30_59']>2).astype(float)+(df['late_90_days']>0).astype(float))
    # NLP stress index (simulated from officer notes sentiment)
    df['note_stress_index']      = np.clip(d*rng.uniform(0.4,0.8,n)+rng.normal(0.1,0.05,n), 0, 1)
    return df

def make_credit(n=32_000, seed=SEED):
    rng = np.random.default_rng(seed)
    y   = (rng.random(n) < 0.22).astype(int)
    grade_map  = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
    home_opts  = ['RENT','MORTGAGE','OWN','OTHER']
    intent_opts= ['PERSONAL','EDUCATION','MEDICAL','VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION']
    df = pd.DataFrame()
    df['loan_status']                = y
    df['person_age']                 = np.clip(rng.normal(28,7,n).astype(int), 18, 80)
    df['person_income']              = np.clip(rng.lognormal(10.7,0.5,n)-y*rng.uniform(5000,15000,n),4000,6_000_000)
    df['person_emp_length']          = np.clip(rng.exponential(4,n), 0, 41)
    df['loan_amnt']                  = np.clip(rng.lognormal(8.8,0.6,n), 500, 35_000)
    df['loan_int_rate']              = np.clip(rng.normal(11,5,n)+y*rng.uniform(0,5,n), 5.42, 23.22)
    df['loan_percent_income']        = np.clip(df['loan_amnt']/df['person_income'], 0.0, 0.66)
    df['cb_person_cred_hist_length'] = np.clip(rng.normal(5.8,3.5,n).astype(int), 2, 30)
    grades = rng.choice(list(grade_map.keys()), n, p=[0.20,0.25,0.20,0.15,0.10,0.07,0.03])
    df['loan_grade_ord']             = [grade_map[g] for g in grades]
    df['default_flag']               = (rng.random(n)<0.18).astype(int)
    df['note_stress_index']          = np.clip(y*rng.uniform(0.4,0.8,n)+rng.normal(0.1,0.05,n), 0, 1)
    df['person_home_ownership']      = rng.choice(home_opts, n, p=[0.46,0.38,0.10,0.06])
    df['loan_intent']                = rng.choice(intent_opts, n, p=[0.20,0.20,0.18,0.14,0.14,0.14])
    return df

# ==================================================================
#  BUILD DATASETS
# ==================================================================
banner("STEP 1 / 7  --  GENERATING DATASETS")

msme_df  = make_msme(30_000)
cred_df  = make_credit(32_000)

STRUCT = ['revolving_utilization','debt_ratio','late_30_59','late_60_89','late_90_days',
          'open_credit_lines','real_estate_loans','num_dependents','income_stability',
          'gst_compliance_score','emi_delay_count','cashflow_stress_ratio',
          'working_capital_usage','revenue_trend_index','payment_history_score',
          'supplier_payment_risk']
FEAT_NLP = STRUCT + ['note_stress_index']
LR_FEATS = ['revolving_utilization','debt_ratio','late_90_days']

NUM_COLS = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate',
            'loan_percent_income','cb_person_cred_hist_length','loan_grade_ord',
            'default_flag','note_stress_index']
CAT_COLS = ['person_home_ownership','loan_intent']

X_m  = msme_df[STRUCT].fillna(msme_df[STRUCT].median())
X_mn = msme_df[FEAT_NLP].fillna(msme_df[FEAT_NLP].median())
y_m  = msme_df['default_flag'].values

X_c  = cred_df[NUM_COLS + CAT_COLS]
y_c  = cred_df['loan_status'].values

Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(X_m,  y_m, test_size=0.25, random_state=SEED, stratify=y_m)
Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(X_mn, y_m, test_size=0.25, random_state=SEED, stratify=y_m)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c,  y_c, test_size=0.20, random_state=SEED, stratify=y_c)

imp_m = SimpleImputer(strategy='median').fit(Xm_tr)
imp_n = SimpleImputer(strategy='median').fit(Xn_tr)
Xm_tr_i, Xm_te_i = imp_m.transform(Xm_tr), imp_m.transform(Xm_te)
Xn_tr_i, Xn_te_i = imp_n.transform(Xn_tr), imp_n.transform(Xn_te)

print(f"  MSME dataset   : {len(X_m):,} samples | default {y_m.mean()*100:.1f}% | "
      f"train {len(Xm_tr):,} / test {len(Xm_te):,}")
print(f"  Credit dataset : {len(X_c):,} samples | default {y_c.mean()*100:.1f}% | "
      f"train {len(Xc_tr):,} / test {len(Xc_te):,}")

spw_m = float((ym_tr==0).sum()/(ym_tr==1).sum())
spw_c = float((yc_tr==0).sum()/(yc_tr==1).sum())

# ==================================================================
#  TRAIN ALL 6 MODELS
# ==================================================================
banner("STEP 2 / 7  --  TRAINING ALL 6 MODELS")
results = {}
timings = {}

# -----------------------------------------------------------------
# MODEL 1 -- Logistic Regression Baseline (3 features)
# -----------------------------------------------------------------
section("Model 1 -- Logistic Regression Baseline  (3 features)")
t0 = time.time()
lr_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler()),
    ('lr',  LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED)),
])
lr_pipe.fit(Xm_tr[LR_FEATS], ym_tr)
timings['LR Baseline'] = time.time() - t0
print(f"  Trained in {timings['LR Baseline']:.1f}s")

# -----------------------------------------------------------------
# MODEL 2 -- XGBoost MSME (16 structured features)
# -----------------------------------------------------------------
section("Model 2 -- XGBoost MSME Engine  (16 structured features)")
t0 = time.time()
xgb_msme = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.85, reg_lambda=2, reg_alpha=0.5,
    scale_pos_weight=spw_m, eval_metric='auc', random_state=SEED, n_jobs=-1)
xgb_msme.fit(Xm_tr_i, ym_tr, eval_set=[(Xm_te_i, ym_te)], verbose=False)
timings['XGBoost MSME'] = time.time() - t0
print(f"  Trained in {timings['XGBoost MSME']:.1f}s  ({xgb_msme.n_estimators} trees)")

# -----------------------------------------------------------------
# MODEL 3 -- XGBoost + NLP Stress Index (17 features)
# -----------------------------------------------------------------
section("Model 3 -- XGBoost + NLP Stress Index  (16 struct + 1 NLP)")
t0 = time.time()
xgb_nlp = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.85, reg_lambda=2, reg_alpha=0.5,
    scale_pos_weight=spw_m, eval_metric='auc', random_state=SEED, n_jobs=-1)
xgb_nlp.fit(Xn_tr_i, yn_tr, eval_set=[(Xn_te_i, yn_te)], verbose=False)
timings['XGBoost + NLP'] = time.time() - t0
print(f"  Trained in {timings['XGBoost + NLP']:.1f}s  ({xgb_nlp.n_estimators} trees)")

# -----------------------------------------------------------------
# MODEL 4 -- LightGBM MSME (16 features)
# -----------------------------------------------------------------
section("Model 4 -- LightGBM MSME Engine  (16 structured features)")
t0 = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.85, reg_lambda=2, reg_alpha=0.5,
    scale_pos_weight=spw_m, random_state=SEED, n_jobs=-1, verbose=-1)
lgb_model.fit(Xm_tr_i, ym_tr,
              eval_set=[(Xm_te_i, ym_te)])
timings['LightGBM'] = time.time() - t0
print(f"  Trained in {timings['LightGBM']:.1f}s")

# -----------------------------------------------------------------
# MODEL 5 -- Optuna-Tuned XGBoost (NLP features, 20 trials)
# -----------------------------------------------------------------
section("Model 5 -- Optuna-Tuned XGBoost  (16+NLP, 20 hyperparameter trials)")
t0 = time.time()
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

def objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 200, 600, step=50),
        max_depth        = trial.suggest_int('max_depth', 3, 7),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        subsample        = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        scale_pos_weight = trial.suggest_float('scale_pos_weight', 1.0, spw_m*1.5),
    )
    m = xgb.XGBClassifier(eval_metric='auc', random_state=SEED, n_jobs=-1, **params)
    return cross_val_score(m, Xn_tr_i, yn_tr, cv=cv, scoring='roc_auc', n_jobs=1).mean()

study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=20)
best = study.best_params
print(f"  Best CV AUC: {study.best_value:.4f}  |  params: {best}")

xgb_tuned = xgb.XGBClassifier(eval_metric='auc', random_state=SEED, n_jobs=-1, **best)
xgb_tuned.fit(Xn_tr_i, yn_tr, verbose=False)

# isotonic calibration
Xtr2, Xcal, ytr2, ycal = train_test_split(Xn_tr_i, yn_tr, test_size=0.2, stratify=yn_tr, random_state=SEED)
cal_m = xgb.XGBClassifier(eval_metric='auc', random_state=SEED, n_jobs=-1, **best).fit(Xtr2, ytr2)
iso   = IsotonicRegression(out_of_bounds='clip').fit(cal_m.predict_proba(Xcal)[:,1], ycal)

timings['XGBoost Tuned'] = time.time() - t0
print(f"  Trained + tuned + calibrated in {timings['XGBoost Tuned']:.1f}s")

# -----------------------------------------------------------------
# MODEL 6 -- XGBoost Real Credit Pipeline (full sklearn Pipeline)
# -----------------------------------------------------------------
section("Model 6 -- XGBoost Real Credit Pipeline  (sklearn Pipeline, num+cat)")
t0 = time.time()
pre = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), NUM_COLS),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
])
real_pipe = Pipeline([
    ('pre', pre),
    ('clf', xgb.XGBClassifier(
        n_estimators=550, max_depth=6, learning_rate=0.057,
        subsample=0.878, colsample_bytree=0.848,
        min_child_weight=10, gamma=3.2,
        reg_lambda=0.069, reg_alpha=0.021,
        scale_pos_weight=spw_c,
        eval_metric='auc', random_state=SEED, n_jobs=-1)),
])
real_pipe.fit(Xc_tr, yc_tr)
timings['XGBoost Real'] = time.time() - t0
print(f"  Trained in {timings['XGBoost Real']:.1f}s")

# ==================================================================
#  SCORE ALL MODELS
# ==================================================================
banner("STEP 3 / 7  --  SCORING ALL MODELS")

p_lr   = lr_pipe.predict_proba(Xm_te[LR_FEATS])[:,1]
p_xm   = xgb_msme.predict_proba(Xm_te_i)[:,1]
p_nlp  = xgb_nlp.predict_proba(Xn_te_i)[:,1]
p_lgb  = lgb_model.predict_proba(Xm_te_i)[:,1]
p_tuned= iso.transform(xgb_tuned.predict_proba(Xn_te_i)[:,1])
p_real = real_pipe.predict_proba(Xc_te)[:,1]

print(f"  Scored all models successfully")

# ==================================================================
#  FULL METRIC REPORTS
# ==================================================================
banner("STEP 4 / 7  --  FULL METRIC REPORTS")

MODEL_CONFIGS = [
    ("Model 1 -- LR Baseline (3 features)         @ thr=0.50", p_lr,    ym_te, 0.50, "LR Baseline"),
    ("Model 2 -- XGBoost MSME (16 features)        @ thr=0.50", p_xm,    ym_te, 0.50, "XGBoost MSME"),
    ("Model 2 -- XGBoost MSME (16 features)        @ thr=0.30", p_xm,    ym_te, 0.30, "XGBoost MSME @0.30"),
    ("Model 3 -- XGBoost + NLP (17 features)       @ thr=0.50", p_nlp,   ym_te, 0.50, "XGBoost+NLP"),
    ("Model 4 -- LightGBM (16 features)            @ thr=0.50", p_lgb,   ym_te, 0.50, "LightGBM"),
    ("Model 5 -- Optuna-Tuned XGBoost (calibrated) @ thr=0.50", p_tuned, ym_te, 0.50, "XGBoost Tuned"),
    ("Model 6 -- Real Credit Pipeline              @ thr=0.50", p_real,  yc_te, 0.50, "XGBoost Real"),
]

for title, probs, y_true, thr, name in MODEL_CONFIGS:
    section(title)
    r = report_metrics(y_true, probs, thr, name)
    results[name] = r

# ==================================================================
#  SUMMARY TABLE
# ==================================================================
banner("STEP 5 / 7  --  SUMMARY TABLE")
print()
hdr = f"  {'Model':<28}  {'Acc%':>6}  {'Prec%':>6}  {'Rec%':>6}  {'F1%':>6}  {'LogLoss':>8}  {'ROC-AUC':>8}"
print(hdr)
print("  " + "-" * (len(hdr)-2))
for name, r in results.items():
    print(f"  {name:<28}  {r['accuracy']*100:6.2f}  {r['precision']*100:6.2f}  "
          f"{r['recall']*100:6.2f}  {r['f1']*100:6.2f}  {r['log_loss']:8.4f}  {r['roc_auc']:8.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\n  Best model by ROC-AUC: [{best_name}]  AUC = {results[best_name]['roc_auc']:.4f}")

# ==================================================================
#  VISUALIZATIONS
# ==================================================================
banner("STEP 6 / 7  --  GENERATING CHARTS")

MODEL_COLORS = {name: COLORS[i % len(COLORS)] for i, name in enumerate(results)}

# -- Fig 1: ROC Curves -------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 7))
fig1.patch.set_facecolor(BG)
for name, r in results.items():
    fpr, tpr, _ = roc_curve(r['y_true'], r['probs'])
    ax1.plot(fpr, tpr, lw=2.5, color=MODEL_COLORS[name],
             label=f"{name}  (AUC={r['roc_auc']:.4f})")
ax1.plot([0,1],[0,1],'w--',alpha=0.3,lw=1,label='Random')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves -- All 6 Models', fontsize=14, fontweight='bold', pad=14)
ax1.legend(loc='lower right', framealpha=0.2, fontsize=8)
ax1.grid(alpha=0.35); ax1.set_xlim([0,1]); ax1.set_ylim([0,1.01])
fig1.tight_layout()
fig1.savefig(os.path.join(REPORT_DIR,'roc_all_models.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  [OK] roc_all_models.png")

# -- Fig 2: Precision-Recall Curves ------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.patch.set_facecolor(BG)
for name, r in results.items():
    prec_v, rec_v, _ = precision_recall_curve(r['y_true'], r['probs'])
    ap = average_precision_score(r['y_true'], r['probs'])
    ax2.plot(rec_v, prec_v, lw=2.5, color=MODEL_COLORS[name],
             label=f"{name}  (AP={ap:.4f})")
ax2.set_xlabel('Recall', fontsize=12); ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curves -- All 6 Models', fontsize=14, fontweight='bold', pad=14)
ax2.legend(loc='upper right', framealpha=0.2, fontsize=8)
ax2.grid(alpha=0.35); ax2.set_xlim([0,1]); ax2.set_ylim([0,1.01])
fig2.tight_layout()
fig2.savefig(os.path.join(REPORT_DIR,'pr_all_models.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  [OK] pr_all_models.png")

# -- Fig 3: Confusion Matrices -----------------------------------
n_cms = len(results)
fig3, axes3 = plt.subplots(1, n_cms, figsize=(5*n_cms, 5))
fig3.patch.set_facecolor(BG)
fig3.suptitle('Confusion Matrices -- All 6 Models', fontsize=14,
              fontweight='bold', color=TXT, y=1.02)
for ax, (name, r) in zip(axes3, results.items()):
    cm_a = np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
    im = ax.imshow(cm_a, cmap='RdPu', aspect='auto')
    lbls = [['TN','FP'],['FN','TP']]
    for i in range(2):
        for j in range(2):
            v = cm_a[i,j]
            ax.text(j, i, f"{lbls[i][j]}\n{v:,}", ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if v < cm_a.max()*0.6 else 'black')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred H','Pred D'], fontsize=8)
    ax.set_yticklabels(['Act H','Act D'], fontsize=8)
    ax.set_title(f"{name}\nAcc {r['accuracy']*100:.1f}%  F1 {r['f1']*100:.1f}%",
                 fontsize=9, color=TXT)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig3.tight_layout()
fig3.savefig(os.path.join(REPORT_DIR,'confusion_all_models.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  [OK] confusion_all_models.png")

# -- Fig 4: Metrics grouped bar chart ----------------------------
metrics_k  = ['accuracy','precision','recall','f1','roc_auc']
metrics_lb = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
names      = list(results.keys())
x          = np.arange(len(metrics_k))
bw         = 0.12
fig4, ax4 = plt.subplots(figsize=(16, 6))
fig4.patch.set_facecolor(BG)
for idx, name in enumerate(names):
    vals = [results[name][m] for m in metrics_k]
    bars = ax4.bar(x + idx*bw, vals, width=bw, label=name,
                   color=MODEL_COLORS[name], alpha=0.88, edgecolor='white', linewidth=0.3)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x()+bar.get_width()/2, h+0.004, f'{h:.2f}',
                 ha='center', va='bottom', fontsize=5.5, color=TXT)
ax4.set_xticks(x + bw*(len(names)-1)/2)
ax4.set_xticklabels(metrics_lb, fontsize=11)
ax4.set_ylim(0, 1.15); ax4.set_ylabel('Score (0-1)', fontsize=11)
ax4.set_title('All Metrics -- All 6 Models', fontsize=14, fontweight='bold', pad=14)
ax4.legend(loc='upper left', framealpha=0.2, fontsize=8, ncol=3)
ax4.grid(axis='y', alpha=0.35)
fig4.tight_layout()
fig4.savefig(os.path.join(REPORT_DIR,'metrics_all_models.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)
print("  [OK] metrics_all_models.png")

# -- Fig 5: Log Loss bar chart -----------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 5))
fig5.patch.set_facecolor(BG)
ll_vals = [results[n]['log_loss'] for n in names]
bars5 = ax5.barh(names, ll_vals, color=[MODEL_COLORS[n] for n in names],
                 alpha=0.85, edgecolor='white', linewidth=0.4)
for bar, val in zip(bars5, ll_vals):
    ax5.text(val+0.003, bar.get_y()+bar.get_height()/2, f'{val:.4f}',
             va='center', fontsize=9.5, color=TXT)
ax5.set_xlabel('Log Loss  (lower is better)', fontsize=11)
ax5.set_title('Log Loss -- All 6 Models', fontsize=13, fontweight='bold', pad=12)
ax5.grid(axis='x', alpha=0.3); ax5.set_xlim(0, max(ll_vals)*1.3)
fig5.tight_layout()
fig5.savefig(os.path.join(REPORT_DIR,'logloss_all_models.png'), dpi=150, bbox_inches='tight')
plt.close(fig5)
print("  [OK] logloss_all_models.png")

# -- Fig 6: Score Distributions (3 key models) -------------------
key_models = ['LR Baseline','XGBoost MSME','XGBoost Tuned']
fig6, axes6 = plt.subplots(1, 3, figsize=(16, 5))
fig6.patch.set_facecolor(BG)
fig6.suptitle('Score Distributions -- Healthy vs Default', fontsize=13, fontweight='bold', color=TXT)
for ax, name in zip(axes6, key_models):
    r = results[name]
    ax.hist(r['probs'][r['y_true']==0], bins=50, alpha=0.55, color='#60CFFF', label='Healthy', density=True)
    ax.hist(r['probs'][r['y_true']==1], bins=50, alpha=0.65, color='#FF6060', label='Default', density=True)
    ax.set_title(name, fontsize=11, color=TXT)
    ax.set_xlabel('Predicted Probability', fontsize=9); ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=9, framealpha=0.2); ax.grid(alpha=0.3)
fig6.tight_layout()
fig6.savefig(os.path.join(REPORT_DIR,'score_dist_all_models.png'), dpi=150, bbox_inches='tight')
plt.close(fig6)
print("  [OK] score_dist_all_models.png")

# -- Fig 7: Master dashboard -------------------------------------
fig7 = plt.figure(figsize=(22, 14))
fig7.patch.set_facecolor(BG)
gs = GridSpec(2, 3, figure=fig7, hspace=0.45, wspace=0.35)

ax_roc = fig7.add_subplot(gs[0,0])
for name, r in results.items():
    fpr, tpr, _ = roc_curve(r['y_true'], r['probs'])
    ax_roc.plot(fpr, tpr, lw=2, color=MODEL_COLORS[name],
                label=f"{name[:16]} ({r['roc_auc']:.3f})")
ax_roc.plot([0,1],[0,1],'w--',alpha=0.25,lw=1)
ax_roc.set_title('ROC Curves', fontweight='bold')
ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
ax_roc.legend(fontsize=6, framealpha=0.2); ax_roc.grid(alpha=0.3)

ax_pr = fig7.add_subplot(gs[0,1])
for name, r in results.items():
    pv, rv, _ = precision_recall_curve(r['y_true'], r['probs'])
    ax_pr.plot(rv, pv, lw=2, color=MODEL_COLORS[name], label=name[:16])
ax_pr.set_title('PR Curves', fontweight='bold')
ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
ax_pr.legend(fontsize=6, framealpha=0.2); ax_pr.grid(alpha=0.3)

ax_bar = fig7.add_subplot(gs[0,2])
x2 = np.arange(len(metrics_k)); bw2 = 0.12
for idx, name in enumerate(names):
    vals = [results[name][m] for m in metrics_k]
    ax_bar.bar(x2+idx*bw2, vals, width=bw2, label=name[:12], color=MODEL_COLORS[name], alpha=0.85)
ax_bar.set_xticks(x2+bw2*(len(names)-1)/2)
ax_bar.set_xticklabels(metrics_lb, fontsize=7, rotation=15)
ax_bar.set_ylim(0,1.12); ax_bar.set_ylabel('Score')
ax_bar.set_title('All Metrics', fontweight='bold')
ax_bar.legend(fontsize=6, framealpha=0.2, ncol=2); ax_bar.grid(axis='y', alpha=0.3)

disp_models = ['LR Baseline','XGBoost MSME','XGBoost Tuned']
for ci, name in enumerate(disp_models):
    r = results[name]
    ax_d = fig7.add_subplot(gs[1, ci])
    ax_d.hist(r['probs'][r['y_true']==0], bins=40, alpha=0.5, color='#60CFFF', label='Healthy', density=True)
    ax_d.hist(r['probs'][r['y_true']==1], bins=40, alpha=0.65, color='#FF6060', label='Default', density=True)
    ax_d.set_title(f'Score Dist -- {name}', fontweight='bold', fontsize=9)
    ax_d.set_xlabel('Probability'); ax_d.set_ylabel('Density')
    ax_d.legend(fontsize=8, framealpha=0.2); ax_d.grid(alpha=0.3)

fig7.suptitle('IDBI MSME Risk Engine -- Complete Model Evaluation Dashboard (All 6 Models)',
              fontsize=15, fontweight='bold', color=TXT, y=1.01)
fig7.savefig(os.path.join(REPORT_DIR,'master_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close(fig7)
print("  [OK] master_dashboard.png")

# ==================================================================
#  SAVE JSON REPORT
# ==================================================================
def serialise(r):
    return {k: (float(v) if isinstance(v,(np.floating,float)) else
                int(v)   if isinstance(v,(np.integer,int))  else v)
            for k,v in r.items() if k not in ('probs','y_true')}

json_report = {n: serialise(r) for n,r in results.items()}
rp = os.path.join(REPORT_DIR,'all_models_results.json')
with open(rp,'w') as f: json.dump(json_report, f, indent=2)
print(f"  [OK] all_models_results.json -> {rp}")

# ==================================================================
#  FINAL SUMMARY
# ==================================================================
banner("STEP 7 / 7  --  DONE")
print()
print(f"  {'Model':<28}  {'Acc%':>6}  {'F1%':>6}  {'Rec%':>6}  {'LogLoss':>8}  {'ROC-AUC':>8}  {'Train(s)':>9}")
print("  " + "-" * 80)
timing_keys = {'LR Baseline':'LR Baseline','XGBoost MSME':'XGBoost MSME',
               'XGBoost MSME @0.30':'XGBoost MSME','XGBoost+NLP':'XGBoost + NLP',
               'LightGBM':'LightGBM','XGBoost Tuned':'XGBoost Tuned','XGBoost Real':'XGBoost Real'}
for name, r in results.items():
    tk = timing_keys.get(name, name)
    t = timings.get(tk, 0)
    print(f"  {name:<28}  {r['accuracy']*100:6.2f}  {r['f1']*100:6.2f}  "
          f"{r['recall']*100:6.2f}  {r['log_loss']:8.4f}  {r['roc_auc']:8.4f}  {t:9.1f}s")

print()
print("  Charts saved to:", REPORT_DIR)
print("    roc_all_models.png      |  pr_all_models.png")
print("    confusion_all_models.png|  metrics_all_models.png")
print("    logloss_all_models.png  |  score_dist_all_models.png")
print("    master_dashboard.png    |  all_models_results.json")
print()
print(f"  Best model: [{best_name}]  ROC-AUC = {results[best_name]['roc_auc']:.4f}")
print()
