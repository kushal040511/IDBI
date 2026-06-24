"""
Model Evaluation — IDBI MSME Risk Engine
=========================================
Evaluates the saved XGBoost model (trained on Give Me Some Credit) against the
labeled synthetic MSME portfolio (data/msme_synthetic_data.csv) — an out-of-sample
test in the model's actual target domain.

Reuses the FastAPI app's own feature-mapping (main.map_upload_to_portfolio) so the
evaluation matches exactly how the running service scores uploaded borrowers.

Reports: Accuracy, Precision, Recall, F1, Log Loss, ROC-AUC, Confusion Matrix
at both the standard 0.50 threshold and the model's 0.30 early-warning threshold.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, roc_auc_score, confusion_matrix)

import main  # reuse the app's feature mapping + globals

DATA = 'data/msme_synthetic_data.csv'

# Wire up the globals the mapping/predict helpers expect (normally set on app startup)
main.FEATURES   = joblib.load('models/feature_list.joblib')
main.imputer    = joblib.load('models/imputer.joblib')
main.xgb_model  = joblib.load('models/xgb_model.joblib')

print("=" * 60)
print("  MODEL EVALUATION — XGBoost MSME Risk Engine")
print("=" * 60)

# 1) Load labeled data; the `default` label is constant per company
raw = pd.read_csv(DATA)
labels = raw.groupby('company_id')['default'].first()
n_companies = labels.shape[0]
print(f"\nTest set:  {n_companies} companies  ({len(raw):,} monthly rows aggregated)")
print(f"Class balance:  defaults {int(labels.sum())} ({labels.mean()*100:.1f}%)  |  "
      f"healthy {int((labels == 0).sum())} ({(labels == 0).mean()*100:.1f}%)")

# 2) Map to model features the same way the API does, then score
mapped = main.map_upload_to_portfolio(raw)
y_true = labels.loc[mapped['company_id']].values
probs  = main.predict_portfolio(mapped)

# 3) Threshold-independent metrics
auc = roc_auc_score(y_true, probs)
ll  = log_loss(y_true, probs)

def report(threshold: float):
    y_pred = (probs >= threshold).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n──  Threshold = {threshold:.2f}  " + "─" * 34)
    print(f"  Accuracy    : {acc*100:6.2f}%")
    print(f"  Precision   : {prec*100:6.2f}%")
    print(f"  Recall      : {rec*100:6.2f}%   (defaults caught)")
    print(f"  F1 score    : {f1*100:6.2f}%")
    print(f"  Confusion Matrix")
    print(f"                    Pred Healthy   Pred Default")
    print(f"    Act Healthy  |   {tn:6d}        {fp:6d}      (FP)")
    print(f"    Act Default  |   {fn:6d} (FN)   {tp:6d}      (TP)")
    return dict(threshold=threshold, accuracy=acc, precision=prec,
                recall=rec, f1=f1, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

print(f"\n── Threshold-independent ── ")
print(f"  ROC-AUC     : {auc:6.4f}")
print(f"  Log Loss    : {ll:6.4f}")

r50 = report(0.50)
r30 = report(0.30)   # the model's early-warning operating point

print("\n" + "=" * 60)
print("  NOTE: labels are from the synthetic MSME generator, so this is a")
print("  domain-transfer test, not the original GMSC hold-out fold.")
print("=" * 60)
