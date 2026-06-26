"""
Bootstrap — rebuild models/real/ artifacts with the current installed versions
of scikit-learn and XGBoost so the FastAPI app can start cleanly.

Run ONCE before starting the server:  python bootstrap_models.py
"""
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap

SEED = 42
OUT  = 'models/real'
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  BOOTSTRAPPING models/real/ for current library versions")
print(f"  XGBoost  : {xgb.__version__}")
import sklearn; print(f"  Sklearn  : {sklearn.__version__}")
print("=" * 60)

# ── 1. Synthetic data matching credit_risk_dataset.csv schema ──
print("\n[1/4] Generating synthetic credit data (32,000 borrowers)...")
rng = np.random.default_rng(SEED)
n   = 32_000
y   = (rng.random(n) < 0.22).astype(int)

grade_map  = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
grade_keys = list(grade_map.keys())
home_opts  = ['RENT','MORTGAGE','OWN','OTHER']
intent_opts= ['PERSONAL','EDUCATION','MEDICAL','VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION']

df = pd.DataFrame()
df['loan_status']                = y
df['person_age']                 = np.clip(rng.normal(28,7,n).astype(int),18,80)
df['person_income']              = np.clip(rng.lognormal(10.7,0.5,n)-y*rng.uniform(5000,15000,n),4000,6_000_000)
df['person_emp_length']          = np.clip(rng.exponential(4,n),0,41)
df['loan_amnt']                  = np.clip(rng.lognormal(8.8,0.6,n),500,35_000)
df['loan_int_rate']              = np.clip(rng.normal(11,5,n)+y*rng.uniform(0,5,n),5.42,23.22)
df['loan_percent_income']        = np.clip(df['loan_amnt']/df['person_income'],0.0,0.66)
df['cb_person_cred_hist_length'] = np.clip(rng.normal(5.8,3.5,n).astype(int),2,30)
grades = rng.choice(grade_keys,n,p=[0.20,0.25,0.20,0.15,0.10,0.07,0.03])
df['loan_grade_ord']             = [grade_map[g] for g in grades]
df['default_flag']               = (rng.random(n)<0.18).astype(int)
df['note_stress_index']          = np.clip(y*rng.uniform(0.4,0.8,n)+rng.normal(0.1,0.05,n),0,1)
df['person_home_ownership']      = rng.choice(home_opts,n,p=[0.46,0.38,0.10,0.06])
df['loan_intent']                = rng.choice(intent_opts,n,p=[0.20,0.20,0.18,0.14,0.14,0.14])

NUM_COLS = ['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate',
            'loan_percent_income','cb_person_cred_hist_length','loan_grade_ord',
            'default_flag','note_stress_index']
CAT_COLS = ['person_home_ownership','loan_intent']
RAW_INPUT = NUM_COLS + CAT_COLS

X = df[RAW_INPUT]
y_arr = df['loan_status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y_arr, test_size=0.2,
                                                     stratify=y_arr, random_state=SEED)
print(f"    Train {len(X_train):,} | Test {len(X_test):,} | default {y_arr.mean()*100:.1f}%")

# ── 2. Build + train Pipeline ──────────────────────────────────
print("\n[2/4] Training XGBoost pipeline...")
spw = float((y_train==0).sum()/(y_train==1).sum())
pre = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), NUM_COLS),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
])
pipe = Pipeline([
    ('pre', pre),
    ('clf', xgb.XGBClassifier(
        n_estimators=550, max_depth=6, learning_rate=0.057,
        subsample=0.878, colsample_bytree=0.848,
        min_child_weight=10, gamma=3.2,
        reg_lambda=0.069, reg_alpha=0.021,
        scale_pos_weight=spw,
        eval_metric='auc', random_state=SEED, n_jobs=-1)),
])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)
print(f"    ROC-AUC: {auc:.4f}")

# ── 3. Calibration (isotonic) ──────────────────────────────────
print("\n[3/4] Calibrating probabilities...")
Xtr2, Xcal, ytr2, ycal = train_test_split(X_train, y_train, test_size=0.2,
                                           stratify=y_train, random_state=SEED)
cal_pipe = Pipeline([
    ('pre', ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUM_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
    ])),
    ('clf', xgb.XGBClassifier(
        n_estimators=550, max_depth=6, learning_rate=0.057,
        subsample=0.878, colsample_bytree=0.848,
        min_child_weight=10, gamma=3.2,
        reg_lambda=0.069, reg_alpha=0.021,
        scale_pos_weight=spw,
        eval_metric='auc', random_state=SEED, n_jobs=-1)),
]).fit(Xtr2, ytr2)
iso = IsotonicRegression(out_of_bounds='clip').fit(cal_pipe.predict_proba(Xcal)[:,1], ycal)
print(f"    Calibrator fitted")

# ── 4. SHAP explainer ──────────────────────────────────────────
print("\n[4/4] Building SHAP explainer (sample 1,000 rows)...")
clf = pipe.named_steps['clf']
pre_fitted = pipe.named_steps['pre']
feat_names = list(pre_fitted.get_feature_names_out())
X_test_t   = pre_fitted.transform(X_test)
explainer  = shap.TreeExplainer(clf)
# warm-up on small sample to confirm it works
_ = explainer.shap_values(X_test_t[:100])
print(f"    SHAP explainer ready  |  {len(feat_names)} features")

# ── Save all artifacts ─────────────────────────────────────────
joblib.dump(pipe,      f'{OUT}/pipeline.joblib')
joblib.dump(iso,       f'{OUT}/calibrator.joblib')
joblib.dump(explainer, f'{OUT}/shap_explainer.joblib')

with open(f'{OUT}/feature_names.json','w') as f:
    json.dump(feat_names, f, indent=2)

schema = {
    "raw_numeric": NUM_COLS,
    "raw_structured_numeric": [c for c in NUM_COLS if c!='note_stress_index'],
    "raw_nlp_numeric": ["note_stress_index"],
    "raw_categorical": CAT_COLS,
    "categories": {
        "person_home_ownership": sorted(home_opts),
        "loan_intent": sorted(intent_opts),
    },
    "grade_map": grade_map,
}
with open(f'{OUT}/raw_schema.json','w') as f:
    json.dump(schema, f, indent=2)

perf = {
    "dataset": "Synthetic credit_risk_dataset (bootstrapped for current lib versions)",
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "default_rate": round(float(y_arr.mean()),4),
    "tuned": {"roc_auc": round(auc,4)},
}
with open(f'{OUT}/performance_report.json','w') as f:
    json.dump(perf, f, indent=2)

print("\n" + "=" * 60)
print("  Bootstrap complete!  Artifacts saved to models/real/")
print(f"    pipeline.joblib  |  calibrator.joblib  |  shap_explainer.joblib")
print(f"    feature_names.json  |  raw_schema.json  |  performance_report.json")
print("  You can now run:  uvicorn main:app --reload --port 8000")
print("=" * 60)
