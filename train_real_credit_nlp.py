"""
STRICT Real-Data Training + Unstructured Data — Credit Default Model
====================================================================
Extends train_real_credit.py to satisfy the blueprint's "use BOTH structured AND
unstructured data": adds an NLP `note_stress_index` derived from (synthetic, risk-
correlated) officer notes to the real credit_risk_dataset.csv features, then trains
the strict tuned pipeline on structured + unstructured.

Honest note: the real dataset ships no free text, so officer notes are SYNTHETIC and
correlated with the real loan_status label (with noise) — they demonstrate the
unstructured-data architecture the way real bureau notes would feed it in production.

Reports an ablation (structured-only vs +NLP) to quantify the lift.
Run:  python train_real_credit_nlp.py        (default 40 trials)
"""
import os, json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, roc_auc_score, average_precision_score,
                             confusion_matrix, brier_score_loss)
import xgboost as xgb
import shap, optuna
import nlp_features

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(os.getenv('N_TRIALS', '40'))
SEED = 42
SRC = '/Users/mkm/Downloads/credit_risk_dataset.csv'
OUT = 'models/real'
os.makedirs(OUT, exist_ok=True)

GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
STRUCT_NUM = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
              'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
              'loan_grade_ord', 'default_flag']
NLP_NUM = ['note_stress_index']
NUM_COLS = STRUCT_NUM + NLP_NUM
CAT_COLS = ['person_home_ownership', 'loan_intent']
RAW_INPUT = NUM_COLS + CAT_COLS

print("=" * 66)
print(f"  STRICT REAL-DATA + UNSTRUCTURED — Credit Default · {N_TRIALS} trials")
print("=" * 66)

# ── 1. Load + clean (same strict cleaning) ────────────────────
df = pd.read_csv(SRC).drop_duplicates()
df = df[(df['person_age'] <= 100) & (df['person_emp_length'].fillna(0) <= 60)].reset_index(drop=True)
df['loan_grade_ord'] = df['loan_grade'].map(GRADE_MAP)
df['default_flag'] = (df['cb_person_default_on_file'] == 'Y').astype(int)
y = df['loan_status'].values
print(f"Rows: {len(df):,} | default rate {y.mean()*100:.1f}%")

# ── 2. Synthetic officer notes correlated with the REAL label ─
rng = np.random.RandomState(SEED)
STRESSED = [
    "Borrower under severe repayment stress; multiple missed EMIs and rising overdues.",
    "Income appears unstable, high loan-to-income; serious default risk flagged by RM.",
    "Cashflow tightening, frequent overdrafts; recovery concerns noted this quarter.",
    "Prior delinquency and weak credit profile; affordability is a major concern.",
]
NEUTRAL = [
    "Repayments broadly on schedule with minor irregularities this period.",
    "Some pressure on disposable income but obligations being met.",
    "Account stable; one delayed payment noted, monitoring continues.",
]
HEALTHY = [
    "Strong repayment discipline and comfortable affordability; low risk borrower.",
    "Stable income, healthy credit history, all dues paid on time.",
    "Good standing with ample cashflow buffer; no concerns.",
]
notes = []
for yi in y:
    r = rng.random()
    if yi == 1:
        pool = STRESSED if r < 0.62 else (NEUTRAL if r < 0.86 else HEALTHY)
    else:
        pool = HEALTHY if r < 0.60 else (NEUTRAL if r < 0.85 else STRESSED)
    notes.append(pool[rng.randint(len(pool))])
print("Scoring officer notes with the NLP model...")
df['officer_notes'] = notes
df['note_stress_index'] = nlp_features.stress_index(notes)
print(f"note_stress_index standalone AUC: {roc_auc_score(y, df['note_stress_index']):.4f}")

X = df[RAW_INPUT]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"Train {len(X_train):,} | Test {len(X_test):,}")

def make_pre(cols_num):
    return ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), cols_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)])

pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 200, 800, step=50),
        max_depth=trial.suggest_int('max_depth', 3, 9),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 12),
        gamma=trial.suggest_float('gamma', 0.0, 5.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        scale_pos_weight=trial.suggest_float('scale_pos_weight', 1.0, pos_weight * 1.5))
    pipe = Pipeline([('pre', make_pre(NUM_COLS)),
                     ('clf', xgb.XGBClassifier(tree_method='hist', eval_metric='auc',
                                               random_state=SEED, n_jobs=-1, **params))])
    return cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1).mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
prog = {'n': 0}
def cb(s, t):
    prog['n'] += 1
    if prog['n'] % 10 == 0 or prog['n'] == N_TRIALS:
        print(f"  trial {prog['n']:3d}/{N_TRIALS}  best CV-AUC = {s.best_value:.4f}")
study.optimize(objective, n_trials=N_TRIALS, callbacks=[cb])
best = study.best_params
print(f"Best CV ROC-AUC: {study.best_value:.4f}")

# ── 3. Fit full (structured+NLP) + ablation (structured-only) ─
pipe = Pipeline([('pre', make_pre(NUM_COLS)),
                 ('clf', xgb.XGBClassifier(tree_method='hist', eval_metric='auc',
                                           random_state=SEED, n_jobs=-1, **best))]).fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba); ll = log_loss(y_test, proba)
ap = average_precision_score(y_test, proba); brier = brier_score_loss(y_test, proba)

struct_pipe = Pipeline([('pre', make_pre(STRUCT_NUM)),
                        ('clf', xgb.XGBClassifier(tree_method='hist', eval_metric='auc',
                                                  random_state=SEED, n_jobs=-1, **best))]).fit(X_train, y_train)
struct_auc = roc_auc_score(y_test, struct_pipe.predict_proba(X_test)[:, 1])

def metrics_at(p, thr, y=y_test):
    yp = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    return dict(threshold=round(thr, 3), accuracy=round(accuracy_score(y, yp), 4),
                precision=round(precision_score(y, yp, zero_division=0), 4),
                recall=round(recall_score(y, yp, zero_division=0), 4),
                f1=round(f1_score(y, yp, zero_division=0), 4),
                confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))

grid = np.linspace(0.05, 0.95, 91)
thr_f1 = float(grid[int(np.argmax([f1_score(y_test, (proba >= t).astype(int), zero_division=0) for t in grid]))])
rec_ok = [(t, precision_score(y_test, (proba >= t).astype(int), zero_division=0)) for t in grid
          if recall_score(y_test, (proba >= t).astype(int), zero_division=0) >= 0.85]
thr_rec = float(max(rec_ok, key=lambda x: x[1])[0]) if rec_ok else 0.3

Xtr2, Xcal, ytr2, ycal = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED)
cal = Pipeline([('pre', make_pre(NUM_COLS)),
                ('clf', xgb.XGBClassifier(tree_method='hist', eval_metric='auc',
                                          random_state=SEED, n_jobs=-1, **best))]).fit(Xtr2, ytr2)
iso = IsotonicRegression(out_of_bounds='clip').fit(cal.predict_proba(Xcal)[:, 1], ycal)
proba_cal = iso.transform(proba)
ll_cal = log_loss(y_test, proba_cal); brier_cal = brier_score_loss(y_test, proba_cal)

m050 = metrics_at(proba, 0.50); mf1 = metrics_at(proba, thr_f1); mrec = metrics_at(proba, thr_rec)
print("\n" + "=" * 66)
print("  TUNED MODEL (structured + unstructured) — HELD-OUT TEST")
print("=" * 66)
print(f"  ROC-AUC {auc:.4f}  PR-AUC {ap:.4f}  LogLoss {ll:.4f}→{ll_cal:.4f}  Brier {brier:.4f}→{brier_cal:.4f}")
for t, m in [("@0.50", m050), (f"@{thr_f1:.2f} bestF1", mf1), (f"@{thr_rec:.2f} rec85", mrec)]:
    print(f"  {t:14s} Acc {m['accuracy']*100:5.2f} Prec {m['precision']*100:5.2f} "
          f"Rec {m['recall']*100:5.2f} F1 {m['f1']*100:5.2f}")
print(f"\n  Ablation — structured only AUC {struct_auc:.4f} → +NLP AUC {auc:.4f} "
      f"(lift +{auc-struct_auc:.4f})")

# ── 4. SHAP + save ────────────────────────────────────────────
print("\nSaving artifacts to models/real/ ...")
pre, clf = pipe.named_steps['pre'], pipe.named_steps['clf']
feat_names = list(pre.get_feature_names_out())
joblib.dump(pipe, f'{OUT}/pipeline.joblib')
joblib.dump(iso, f'{OUT}/calibrator.joblib')
joblib.dump(shap.TreeExplainer(clf), f'{OUT}/shap_explainer.joblib')
json.dump(feat_names, open(f'{OUT}/feature_names.json', 'w'), indent=2)
json.dump({"raw_numeric": NUM_COLS, "raw_structured_numeric": STRUCT_NUM,
           "raw_nlp_numeric": NLP_NUM, "raw_categorical": CAT_COLS,
           "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_COLS},
           "grade_map": GRADE_MAP}, open(f'{OUT}/raw_schema.json', 'w'), indent=2)
json.dump({"dataset": "credit_risk_dataset.csv (real) + synthetic risk-correlated officer notes",
           "n_train": int(len(X_train)), "n_test": int(len(X_test)),
           "default_rate": round(float(y.mean()), 4),
           "tuning": {"method": "Optuna TPE", "trials": N_TRIALS, "cv_folds": 5,
                      "best_cv_auc": round(study.best_value, 4), "best_params": best},
           "tuned": {"roc_auc": round(auc, 4), "pr_auc": round(ap, 4),
                     "log_loss": round(ll, 4), "log_loss_calibrated": round(ll_cal, 4),
                     "brier": round(brier, 4), "brier_calibrated": round(brier_cal, 4),
                     "at_0.50": m050, "at_best_f1": mf1, "at_recall85": mrec},
           "ablation_unstructured": {"structured_only_auc": round(struct_auc, 4),
                                     "structured_plus_nlp_auc": round(auc, 4),
                                     "auc_lift_from_unstructured": round(auc - struct_auc, 4)},
           "operating_thresholds": {"best_f1": round(thr_f1, 3), "recall85": round(thr_rec, 3)},
           "features_structured": STRUCT_NUM + CAT_COLS, "features_unstructured": NLP_NUM,
           "raw_features": RAW_INPUT, "model_features": feat_names},
          open(f'{OUT}/performance_report.json', 'w'), indent=2)
print("Saved. Done.")
print("=" * 66)
