"""
IDBI Credit Risk Intelligence — FastAPI Backend (REAL-DATA MODEL)
=================================================================
Runs the STRICT model trained on credit_risk_dataset.csv (32.5k real borrowers,
native features, Optuna-tuned, calibrated). See train_real_credit.py.

The displayed demo portfolio is SYNTHETIC (generated in the real credit schema) —
the real dataset rows are used only for training, never displayed.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import pandas as pd
import numpy as np
import joblib
import os
import io
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import nlp_features  # unstructured-data (officer notes) → note_stress_index

app = FastAPI(title="Credit Risk Intelligence API")
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="static")

# ─── Globals ─────────────────────────────────────────────────
pipeline = None        # full sklearn Pipeline (preprocess + XGBoost)
calibrator = None      # isotonic calibrator
explainer = None       # SHAP TreeExplainer on the XGB step
FEAT_NAMES = None      # transformed feature names (20)
SCHEMA = None          # raw input schema
PERF = {}
feature_df = None

RAW_STRUCT_NUM = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                  'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                  'loan_grade_ord', 'default_flag']
RAW_NLP_NUM = ['note_stress_index']               # unstructured-data feature
RAW_NUM = RAW_STRUCT_NUM + RAW_NLP_NUM
RAW_CAT = ['person_home_ownership', 'loan_intent']
RAW_INPUT = RAW_NUM + RAW_CAT

# Blueprint risk-factor groupings (for the borrower-detail breakdown)
RISK_FACTOR_GROUPS = {
    "Repayment & Bureau": ['default_flag', 'cb_person_cred_hist_length', 'loan_grade_ord'],
    "Affordability / Cashflow": ['loan_percent_income', 'loan_int_rate', 'loan_amnt'],
    "Borrower Profile": ['person_age', 'person_income', 'person_emp_length'],
    "Unstructured (Officer Notes)": ['note_stress_index'],
}
GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
GRADE_LETTER = {v: k for k, v in GRADE_MAP.items()}
INTENTS = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
HOME = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']

# LGD keyed by loan purpose (shown as "sector" in the dashboard)
LGD_MAP = {'DEBTCONSOLIDATION': 0.45, 'EDUCATION': 0.40, 'HOMEIMPROVEMENT': 0.35,
           'MEDICAL': 0.50, 'PERSONAL': 0.45, 'VENTURE': 0.55}

ACTION_LADDER = {
    'Low': [{"priority": "routine", "action": "Continue Standard Monitoring",
             "detail": "No immediate action required. Review at next scheduled cycle."}],
    'Medium': [{"priority": "medium", "action": "Schedule Relationship Manager Call",
                "detail": "Discuss repayment capacity with borrower within 7 days."},
               {"priority": "medium", "action": "Request Updated Income Proof",
                "detail": "Obtain recent salary slips / bank statements."}],
    'High': [{"priority": "high", "action": "Initiate Document Review",
              "detail": "Verify income, employment, and collateral documents."},
             {"priority": "high", "action": "Affordability Re-assessment",
              "detail": "Re-check loan-to-income and obligations vs current income."},
             {"priority": "high", "action": "Restrict New Exposure",
              "detail": "Hold any top-up / additional credit until reviewed."}],
    'Critical': [{"priority": "critical", "action": "Initiate Restructuring Discussion",
                  "detail": "Escalate to Senior RM for restructuring evaluation."},
                 {"priority": "critical", "action": "Field Investigation",
                  "detail": "Dispatch officer to verify borrower status."},
                 {"priority": "critical", "action": "Assign to Recovery Monitoring",
                  "detail": "Flag in Recovery system and assign recovery officer."},
                 {"priority": "critical", "action": "Prepare Legal Documentation",
                  "detail": "Engage legal team for NPA declaration / notices."}],
}


def get_risk_band(pd_value: float) -> str:
    if pd_value < 0.20: return "Low"
    if pd_value < 0.50: return "Medium"
    if pd_value < 0.75: return "High"
    return "Critical"


# ─── SHAP narratives (raw + one-hot encoded feature names) ───
def narrate_feature(fname: str, value: float) -> str:
    base = fname.split('__', 1)[-1]
    if fname.startswith('cat__'):
        col, cat = base.split('_', 1) if '_' in base else (base, '')
        if 'home_ownership' in fname:
            return f"Home ownership: {fname.split('_')[-1].title()}." if value > 0.5 else ""
        if 'loan_intent' in fname:
            return f"Loan purpose: {fname.split('loan_intent_')[-1].title()}." if value > 0.5 else ""
        return f"{base}: {'yes' if value > 0.5 else 'no'}."
    # numeric
    if base == 'note_stress_index':    return f"Officer-notes stress index (NLP): {value:.2f}/1.0 — {'alarming language in notes' if value > 0.66 else 'some concern in notes' if value > 0.4 else 'notes read healthy'}."
    if base == 'person_age':           return f"Borrower age: {value:.0f} years."
    if base == 'person_income':        return f"Annual income: ${value:,.0f} — {'low' if value < 40000 else 'moderate' if value < 90000 else 'high'}."
    if base == 'person_emp_length':    return f"Employment length: {value:.0f} year(s) — {'short tenure' if value < 2 else 'stable'}."
    if base == 'loan_amnt':            return f"Loan amount: ${value:,.0f}."
    if base == 'loan_int_rate':        return f"Interest rate: {value:.1f}% — {'high (subprime)' if value > 15 else 'elevated' if value > 11 else 'prime'}."
    if base == 'loan_percent_income':  return f"Loan is {value*100:.0f}% of income — {'very high burden' if value > 0.4 else 'high' if value > 0.25 else 'manageable'}."
    if base == 'cb_person_cred_hist_length': return f"Credit history: {value:.0f} year(s) — {'thin file' if value < 3 else 'established'}."
    if base == 'loan_grade_ord':       return f"Loan grade: {GRADE_LETTER.get(int(round(value)), '?')} ({value:.0f}/7) — {'subprime' if value >= 4 else 'prime'}."
    if base == 'default_flag':         return "Prior default on file — major red flag." if value > 0.5 else "No prior default on file."
    return f"{base}: {value:.2f}"

_STRESSED_NOTES = [
    "Borrower under severe repayment stress; multiple missed EMIs and rising overdues.",
    "Income appears unstable, high loan-to-income; serious default risk flagged by RM.",
    "Cashflow tightening, frequent overdrafts; recovery concerns noted this quarter.",
]
_NEUTRAL_NOTES = [
    "Repayments broadly on schedule with minor irregularities this period.",
    "Some pressure on disposable income but obligations being met.",
]
_HEALTHY_NOTES = [
    "Strong repayment discipline and comfortable affordability; low risk borrower.",
    "Stable income, healthy credit history, all dues paid on time.",
]

def _pick_note(rng, risk_proxy: float) -> str:
    r = rng.random()
    if risk_proxy >= 0.6:
        pool = _STRESSED_NOTES if r < 0.7 else _NEUTRAL_NOTES
    elif risk_proxy >= 0.33:
        pool = _NEUTRAL_NOTES if r < 0.6 else (_STRESSED_NOTES if r < 0.8 else _HEALTHY_NOTES)
    else:
        pool = _HEALTHY_NOTES if r < 0.7 else _NEUTRAL_NOTES
    return pool[rng.randint(len(pool))]


RAW_NARRATIVE = {
    'note_stress_index': lambda v: f"Officer-notes stress index (NLP): {v:.2f}/1.0 — {'alarming language' if v > 0.66 else 'some concern' if v > 0.4 else 'notes read healthy'}.",
    'person_age': lambda v: f"Borrower age: {int(v)} years.",
    'person_income': lambda v: f"Annual income: ${v:,.0f} — {'low' if v < 40000 else 'moderate' if v < 90000 else 'high'}.",
    'person_emp_length': lambda v: f"Employment length: {v:.0f} year(s) — {'short tenure' if v < 2 else 'stable'}.",
    'loan_amnt': lambda v: f"Loan amount: ${v:,.0f}.",
    'loan_int_rate': lambda v: f"Interest rate: {v:.1f}% — {'subprime' if v > 15 else 'elevated' if v > 11 else 'prime'}.",
    'loan_percent_income': lambda v: f"Loan is {v*100:.0f}% of income — {'very high burden' if v > 0.4 else 'high' if v > 0.25 else 'manageable'}.",
    'cb_person_cred_hist_length': lambda v: f"Credit history: {int(v)} year(s).",
    'loan_grade_ord': lambda v: f"Loan grade {GRADE_LETTER.get(int(round(v)), '?')} ({int(v)}/7).",
    'default_flag': lambda v: "Prior default on file." if v > 0.5 else "No prior default.",
}


# ─── Synthetic borrower generation (real credit schema) ──────
def _make_borrower(company_id: str, rng: np.random.RandomState) -> dict:
    grade_ord = int(rng.choice([1, 2, 3, 4, 5, 6, 7], p=[.18, .26, .22, .16, .10, .05, .03]))
    base_rate = {1: 7.5, 2: 11.0, 3: 13.5, 4: 15.3, 5: 16.8, 6: 18.5, 7: 20.2}[grade_ord]
    int_rate = float(np.clip(base_rate + rng.normal(0, 0.8), 5, 24))
    income = float(np.clip(rng.lognormal(11.0, 0.55), 12000, 1_500_000))
    age = int(np.clip(rng.normal(28, 7), 21, 70))
    emp_len = float(np.clip(rng.exponential(4), 0, min(age - 18, 35)))
    cred_hist = int(np.clip(rng.poisson(5) + 2, 2, min(age - 16, 30)))
    loan_amnt = float(np.clip(rng.lognormal(9.1, 0.5), 500, 35000))
    pct_income = float(np.clip(loan_amnt / income, 0.01, 0.83))
    default_flag = int(rng.random() < (0.10 + 0.05 * grade_ord))
    home = str(rng.choice(HOME, p=[.50, .12, .36, .02]))
    intent = str(rng.choice(INTENTS))
    # risk proxy → officer note (unstructured); note_stress_index batch-scored by caller
    risk_proxy = float(np.clip(0.30 * (grade_ord / 7) + 0.30 * min(pct_income / 0.5, 1)
                               + 0.25 * ((int_rate - 7) / 15) + 0.15 * default_flag, 0, 1))
    return {
        'company_id': company_id,
        'person_age': age, 'person_income': round(income, 0),
        'person_emp_length': round(emp_len, 1), 'loan_amnt': round(loan_amnt, 0),
        'loan_int_rate': round(int_rate, 2), 'loan_percent_income': round(pct_income, 3),
        'cb_person_cred_hist_length': cred_hist, 'loan_grade_ord': grade_ord,
        'default_flag': default_flag, 'person_home_ownership': home, 'loan_intent': intent,
        'officer_notes': _pick_note(rng, risk_proxy),
        # display slots reused by the dashboard
        'sector': intent.title(), 'loan_type': f"Grade {GRADE_LETTER[grade_ord]}",
        'outstanding_loan': round(loan_amnt * 100, 0),   # scaled to ₹ for display
        'journey_events': '[]',
    }


def build_portfolio() -> pd.DataFrame:
    rng = np.random.RandomState(2024)
    df = pd.DataFrame([_make_borrower(f"BR-{i+1:04d}", rng) for i in range(200)])
    df['note_stress_index'] = nlp_features.stress_index(df['officer_notes'])
    return df


_seq = 0
def generate_new_loans(count: int) -> pd.DataFrame:
    global _seq
    rng = np.random.RandomState()
    rows = []
    for _ in range(count):
        _seq += 1
        rows.append(_make_borrower(f"BR-N{_seq:04d}", rng))
    df = pd.DataFrame(rows)
    df['note_stress_index'] = nlp_features.stress_index(df['officer_notes'])
    return df


def _intent_to_display(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'sector' not in df.columns:
        df['sector'] = df['loan_intent'].str.title()
    if 'loan_type' not in df.columns:
        df['loan_type'] = df['loan_grade_ord'].map(lambda g: f"Grade {GRADE_LETTER.get(int(g), '?')}")
    if 'outstanding_loan' not in df.columns:
        df['outstanding_loan'] = df['loan_amnt'] * 100
    if 'journey_events' not in df.columns:
        df['journey_events'] = '[]'
    return df


def predict_portfolio(df: pd.DataFrame) -> np.ndarray:
    raw = df[RAW_INPUT]
    p = pipeline.predict_proba(raw)[:, 1]
    if calibrator is not None:
        p = calibrator.transform(p)
    return p


@app.on_event("startup")
def load_assets():
    global pipeline, calibrator, explainer, FEAT_NAMES, SCHEMA, PERF, feature_df
    pipeline = joblib.load('models/real/pipeline.joblib')
    try:
        calibrator = joblib.load('models/real/calibrator.joblib')
    except Exception:
        calibrator = None
    explainer = joblib.load('models/real/shap_explainer.joblib')
    FEAT_NAMES = json.load(open('models/real/feature_names.json'))
    SCHEMA = json.load(open('models/real/raw_schema.json'))
    try:
        PERF = json.load(open('models/real/performance_report.json'))
    except Exception:
        PERF = {}
    feature_df = build_portfolio()
    print(f"Real credit model loaded. Portfolio: {len(feature_df)} borrowers.")


# ─── Routes ──────────────────────────────────────────────────
@app.get("/")
def root():
    return RedirectResponse(url="/dashboard/index.html")


@app.get("/model-performance")
def model_performance():
    return PERF


@app.get("/portfolio")
def portfolio_summary():
    if feature_df is None:
        raise HTTPException(503, "Model not loaded")
    probs = predict_portfolio(feature_df)
    df = feature_df.copy()
    df['pd'] = probs
    df['band'] = df['pd'].apply(get_risk_band)
    df['lgd'] = df['loan_intent'].map(lambda s: LGD_MAP.get(s, 0.45))
    df['el'] = df['outstanding_loan'] * df['pd'] * df['lgd']

    sector_stats = []
    for sector, grp in df.groupby('sector'):
        sector_stats.append({
            "sector": sector, "exposure": float(grp['outstanding_loan'].sum()),
            "avg_pd": float(grp['pd'].mean()), "expected_loss": float(grp['el'].sum()),
            "borrower_count": int(len(grp))})
    dist = df['band'].value_counts().to_dict()
    for b in ["Low", "Medium", "High", "Critical"]:
        dist.setdefault(b, 0)
    return {
        "total_borrowers": int(len(df)), "avg_pd": float(probs.mean()),
        "total_exposure": float(df['outstanding_loan'].sum()),
        "total_expected_loss": float(df['el'].sum()),
        "risk_distribution": dist,
        "sector_analytics": sorted(sector_stats, key=lambda x: x['expected_loss'], reverse=True)}


@app.get("/borrowers")
def borrowers(limit: int = 50):
    if feature_df is None:
        raise HTTPException(503, "Model not loaded")
    df = feature_df.copy()
    df['pd'] = predict_portfolio(feature_df)
    top = df.sort_values('pd', ascending=False).head(limit)
    return [{"company_id": r['company_id'], "sector": r['sector'], "loan_type": r['loan_type'],
             "pd": float(r['pd']), "risk_score": int(round(float(r['pd']) * 100)),
             "outstanding_loan": float(r['outstanding_loan']),
             "risk_band": get_risk_band(float(r['pd']))} for _, r in top.iterrows()]


@app.get("/borrowers/{company_id}")
def borrower_details(company_id: str):
    if feature_df is None:
        raise HTTPException(503, "Model not loaded")
    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(404, "Borrower not found")
    row = row.iloc[0]
    base = row[RAW_INPUT].to_dict()
    current_pd = float(predict_portfolio(pd.DataFrame([row[RAW_INPUT].to_dict()]))[0])

    # Simulated 12-month PD trajectory: rising rate pressure + loan-to-income drift
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    timeline = []
    for i, m in enumerate(months):
        scale = i / 11
        f = dict(base)
        f['loan_int_rate'] = min(24, base['loan_int_rate'] * (0.95 + 0.18 * scale))
        f['loan_percent_income'] = min(0.83, base['loan_percent_income'] * (0.9 + 0.4 * scale))
        f['person_emp_length'] = max(0, base['person_emp_length'] * (1.0 - 0.1 * scale))
        mp = float(predict_portfolio(pd.DataFrame([f]))[0])
        timeline.append({"month": m, "pd": mp})

    intent = row['loan_intent']
    lgd = LGD_MAP.get(intent, 0.45)
    ead = float(row['outstanding_loan'])
    el = current_pd * ead * lgd
    note = str(row.get('officer_notes', '') or '')
    nsi = float(row['note_stress_index']) if 'note_stress_index' in row else None
    # Blueprint risk-factor breakdown (grouped)
    risk_factors = {}
    for group, cols in RISK_FACTOR_GROUPS.items():
        risk_factors[group] = [{"feature": c, "value": float(row[c]),
                                "narrative": RAW_NARRATIVE.get(c, lambda v: f"{c}: {v:.2f}")(float(row[c]))}
                               for c in cols if c in row]
    return {
        "company_id": company_id, "sector": row['sector'], "loan_type": row['loan_type'],
        "outstanding_loan": ead, "lgd_pct": lgd, "expected_loss": el,
        "potential_recovery": ead - el, "current_pd": current_pd,
        "risk_score": int(round(current_pd * 100)),
        "risk_band": get_risk_band(current_pd), "timeline": timeline,
        "risk_migration": {"start_band": get_risk_band(timeline[0]['pd']),
                           "end_band": get_risk_band(current_pd)},
        "journey_events": [], "officer_notes": note, "note_stress_index": nsi,
        "risk_factors": risk_factors,
        "raw_features": {k: float(row[k]) for k in RAW_NUM},
        "action_ladder": ACTION_LADDER.get(get_risk_band(current_pd), [])}


@app.get("/borrowers/{company_id}/explain")
def explain(company_id: str):
    if feature_df is None or explainer is None:
        raise HTTPException(503, "Model not loaded")
    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(404, "Borrower not found")
    raw = row.iloc[[0]][RAW_INPUT]
    Xt = pipeline.named_steps['pre'].transform(raw)
    sv = explainer.shap_values(Xt)
    vals = np.asarray(Xt)[0]
    drivers = []
    for i, fn in enumerate(FEAT_NAMES):
        narr = narrate_feature(fn, float(vals[i]))
        if not narr:
            continue
        drivers.append({"feature": fn, "value": float(vals[i]),
                        "impact": float(sv[0][i]), "narrative": narr})
    drivers.sort(key=lambda d: abs(d['impact']), reverse=True)
    return {"key_drivers": drivers[:8]}


# ─── Upload (credit_risk schema CSV) ─────────────────────────
@app.post("/upload")
async def upload_portfolio(file: UploadFile = File(...)):
    global feature_df
    if pipeline is None:
        raise HTTPException(503, "Model not loaded")
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, "Please upload a .csv file")
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    df.columns = [c.strip() for c in df.columns]
    # derive encoded helpers if raw credit_risk columns present
    if 'loan_grade' in df.columns and 'loan_grade_ord' not in df.columns:
        df['loan_grade_ord'] = df['loan_grade'].map(GRADE_MAP)
    if 'cb_person_default_on_file' in df.columns and 'default_flag' not in df.columns:
        df['default_flag'] = (df['cb_person_default_on_file'] == 'Y').astype(int)
    # Unstructured: score officer notes → note_stress_index (neutral 0.5 if no notes column)
    if 'note_stress_index' not in df.columns:
        if 'officer_notes' in df.columns:
            df['note_stress_index'] = nlp_features.stress_index(df['officer_notes'])
        else:
            df['note_stress_index'] = 0.5
    missing = [c for c in RAW_INPUT if c not in df.columns]
    if missing:
        raise HTTPException(422, f"CSV missing required columns: {missing}")
    if 'company_id' not in df.columns:
        df['company_id'] = [f"UP-{i+1:04d}" for i in range(len(df))]
    df = _intent_to_display(df)
    try:
        probs = predict_portfolio(df)
    except Exception as e:
        raise HTTPException(422, f"Could not score uploaded rows: {e}")
    feature_df = df
    return {"status": "ok", "filename": file.filename, "rows_ingested": int(len(df)),
            "borrowers_scored": int(len(df)), "avg_pd": float(probs.mean()),
            "high_risk_count": int((probs >= 0.5).sum())}


@app.post("/reset")
def reset_portfolio():
    global feature_df
    feature_df = build_portfolio()
    return {"status": "ok", "borrowers": int(len(feature_df))}


@app.post("/refresh")
def refresh_portfolio(count: int = 10):
    global feature_df
    if feature_df is None:
        raise HTTPException(503, "Model not loaded")
    count = max(1, min(int(count), 100))
    new_df = generate_new_loans(count)
    feature_df = pd.concat([feature_df, new_df], ignore_index=True)
    probs = predict_portfolio(new_df)
    new_loans = sorted([{"company_id": r['company_id'], "sector": r['sector'],
                         "pd": float(p), "risk_band": get_risk_band(float(p)),
                         "outstanding_loan": float(r['outstanding_loan'])}
                        for (_, r), p in zip(new_df.iterrows(), probs)],
                       key=lambda x: x['pd'], reverse=True)
    return {"status": "ok", "added": count, "total_borrowers": int(len(feature_df)),
            "new_high_risk": int((probs >= 0.5).sum()), "new_avg_pd": float(probs.mean()),
            "new_loans": new_loans}


# ─── Agentic Risk Copilot (pluggable: Gemini / Claude / template) ──
def _borrower_context(company_id: str) -> dict:
    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(404, "Borrower not found")
    raw = row.iloc[[0]][RAW_INPUT]
    pd_val = float(predict_portfolio(raw if set(RAW_INPUT).issubset(raw.columns) else row)[0])
    band = get_risk_band(pd_val)
    Xt = pipeline.named_steps['pre'].transform(raw)
    sv = explainer.shap_values(Xt); vals = np.asarray(Xt)[0]
    drivers = sorted([{"feature": fn, "impact": float(sv[0][i]),
                       "narrative": narrate_feature(fn, float(vals[i]))}
                      for i, fn in enumerate(FEAT_NAMES) if narrate_feature(fn, float(vals[i]))],
                     key=lambda d: abs(d['impact']), reverse=True)[:5]
    r = row.iloc[0]
    lgd = LGD_MAP.get(r['loan_intent'], 0.45); ead = float(r['outstanding_loan'])
    return {"company_id": company_id, "sector": r['sector'], "pd": pd_val, "band": band,
            "ead": ead, "lgd": lgd, "expected_loss": pd_val * ead * lgd, "drivers": drivers,
            "officer_notes": str(r.get('officer_notes', '') or ''),
            "note_stress_index": float(r['note_stress_index']) if 'note_stress_index' in r else None,
            "actions": [a["action"] for a in ACTION_LADDER.get(band, [])]}


def _portfolio_attention() -> list:
    df = feature_df.copy(); df['pd'] = predict_portfolio(feature_df)
    top = df.sort_values('pd', ascending=False).head(5)
    return [{"company_id": r['company_id'], "sector": r['sector'], "pd": float(r['pd']),
             "band": get_risk_band(float(r['pd']))} for _, r in top.iterrows()]


def copilot_engine() -> str:
    if os.getenv("ANTHROPIC_API_KEY"): return "claude"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"): return "gemini"
    return "template"


COPILOT_SYSTEM = ("You are a credit risk officer's copilot at a bank. Answer the manager's "
                  "question using ONLY the structured borrower/portfolio context provided. Be "
                  "concise, factual, plain English. Do not invent numbers.")


def _ctx_json(ctx):
    return json.dumps({"portfolio_attention": _portfolio_attention()} if ctx is None else ctx, indent=2)


def _answer_template(q, ctx):
    if ctx is None:
        rows = _portfolio_attention()
        return "Borrowers requiring immediate attention (highest PD):\n" + "\n".join(
            f"• {r['company_id']} ({r['sector']}) — PD {r['pd']*100:.1f}%, {r['band']}" for r in rows)
    drivers = "; ".join(d['narrative'] for d in ctx['drivers'][:3]) or "no dominant signals"
    ql = q.lower()
    if any(w in ql for w in ("action", "do", "recommend")):
        return (f"{ctx['company_id']} is {ctx['band']} (PD {ctx['pd']*100:.1f}%). "
                f"Recommended: {', '.join(ctx['actions']) or 'standard monitoring'}.")
    if any(w in ql for w in ("changed", "trend", "month")):
        return (f"Key deteriorating signals for {ctx['company_id']}: {drivers}. PD now "
                f"{ctx['pd']*100:.1f}% ({ctx['band']}).")
    return (f"{ctx['company_id']} ({ctx['sector']}) carries a {ctx['pd']*100:.1f}% PD "
            f"({ctx['band']}). Key drivers: {drivers}. Expected loss ≈ ₹{ctx['expected_loss']:,.0f} "
            f"(PD × EAD ₹{ctx['ead']:,.0f} × LGD {ctx['lgd']*100:.0f}%).")


def _answer_gemini(q, ctx):
    import time
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    cfg = types.GenerateContentConfig(system_instruction=COPILOT_SYSTEM, max_output_tokens=600,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    contents = f"Context:\n{_ctx_json(ctx)}\n\nManager's question: {q}"
    TRANSIENT = ("ServerError", "ReadError", "ConnectError", "ReadTimeout", "ConnectTimeout",
                 "RemoteProtocolError", "APIError")
    last = None
    for a in range(4):
        try:
            r = client.models.generate_content(model=model, contents=contents, config=cfg)
            if (r.text or "").strip():
                return r.text.strip()
            last = RuntimeError("empty")
        except genai.errors.ClientError:
            raise
        except Exception as e:
            if type(e).__name__ not in TRANSIENT:
                raise
            last = e
        time.sleep(1.2 * (a + 1))
    raise last


def _answer_claude(q, ctx):
    import anthropic
    client = anthropic.Anthropic()
    m = client.messages.create(model="claude-opus-4-8", max_tokens=400, system=COPILOT_SYSTEM,
                               messages=[{"role": "user",
                                          "content": f"Context:\n{_ctx_json(ctx)}\n\nManager's question: {q}"}])
    return "".join(b.text for b in m.content if getattr(b, "type", None) == "text")


@app.post("/copilot")
def copilot(payload: dict = Body(...)):
    if feature_df is None:
        raise HTTPException(503, "Model not loaded")
    q = (payload.get("question") or "").strip()
    if not q:
        raise HTTPException(400, "question is required")
    cid = payload.get("company_id")
    ctx = _borrower_context(cid) if cid else None
    engine = copilot_engine()
    try:
        answer = (_answer_claude(q, ctx) if engine == "claude"
                  else _answer_gemini(q, ctx) if engine == "gemini"
                  else _answer_template(q, ctx))
    except Exception as e:
        answer = _answer_template(q, ctx); engine = f"template (fallback: {type(e).__name__})"
    return {"engine": engine, "company_id": cid, "question": q, "answer": answer}
