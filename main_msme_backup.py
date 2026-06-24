"""
IDBI MSME Risk Intelligence — FastAPI Backend
===============================================
Real XGBoost model trained on Give Me Some Credit (150k borrowers)
with MSME behavioral feature overlays.
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
    load_dotenv()  # load ANTHROPIC_API_KEY (and any other vars) from a local .env file
except ImportError:
    pass

import nlp_features  # unstructured-data (officer notes) → note_stress_index

app = FastAPI(title="MSME Risk Intelligence API")
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="static")

# ─── Global State ────────────────────────────────────────────
xgb_model   = None
explainer   = None
imputer     = None
feature_df  = None  # Synthetic MSME borrower portfolio
FEATURES    = None

# ─── Sector LGD (Loss Given Default) ────────────────────────
LGD_MAP = {
    'Retail':        0.35,
    'Manufacturing': 0.45,
    'Construction':  0.55,
    'Technology':    0.30,
    'Agriculture':   0.40,
    'Textile':       0.42,
}

# ─── Action Escalation Ladder ────────────────────────────────
ACTION_LADDER = {
    'Low': [
        {"priority": "routine", "action": "Continue Standard Monitoring",
         "detail": "No immediate action required. Review at next scheduled cycle."}
    ],
    'Medium': [
        {"priority": "medium", "action": "Schedule Relationship Manager Call",
         "detail": "Discuss business health with borrower within 7 days."},
        {"priority": "medium", "action": "Request Updated Financial Statements",
         "detail": "Obtain last 3 months bank statements and GST returns."}
    ],
    'High': [
        {"priority": "high", "action": "Initiate Document Review",
         "detail": "Pull all collateral documents and verify current valuation."},
        {"priority": "high", "action": "Cashflow Assessment",
         "detail": "Conduct detailed analysis of inflows vs outflows for past 6 months."},
        {"priority": "high", "action": "Collateral Verification",
         "detail": "Dispatch field officer to verify primary collateral on record."}
    ],
    'Critical': [
        {"priority": "critical", "action": "Initiate Restructuring Discussion",
         "detail": "Escalate to Senior RM for restructuring evaluation immediately."},
        {"priority": "critical", "action": "Evaluate Tenure Extension",
         "detail": "Assess feasibility of extending loan tenure to reduce immediate burden."},
        {"priority": "critical", "action": "Assign to Recovery Monitoring",
         "detail": "Flag in Recovery system and assign dedicated recovery officer."},
        {"priority": "critical", "action": "Prepare Legal Documentation",
         "detail": "Engage legal team to prepare NPA declaration and recovery notices."}
    ]
}

# ─── SHAP → Plain English Narratives ─────────────────────────
FEATURE_NARRATIVE = {
    'revolving_utilization':  lambda v: f"Working capital line utilized at {v*100:.0f}% — {'dangerously high' if v > 0.7 else 'elevated' if v > 0.4 else 'healthy'}.",
    'debt_ratio':             lambda v: f"Debt-to-income ratio is {v:.2f} — {'critical' if v > 1.0 else 'high' if v > 0.5 else 'manageable'}.",
    'late_30_59':             lambda v: f"Borrower had {int(v)} payment(s) 30-59 days late in the observation window.",
    'late_60_89':             lambda v: f"Borrower had {int(v)} payment(s) 60-89 days overdue — significant stress indicator.",
    'late_90_days':           lambda v: f"Borrower had {int(v)} serious delinquency event(s) (90+ days late) — NPA risk.",
    'open_credit_lines':      lambda v: f"Borrower holds {int(v)} open credit lines — {'over-leveraged' if v > 15 else 'normal'}.",
    'real_estate_loans':      lambda v: f"Borrower has {int(v)} real estate loan(s) as collateral exposure.",
    'num_dependents':         lambda v: f"Borrower supports {int(v)} dependent(s) — impacts disposable income.",
    'income_stability':       lambda v: f"Income stability index: {v:.2f} — {'strong' if v > 0.6 else 'moderate' if v > 0.3 else 'weak'}.",
    'gst_compliance_score':   lambda v: f"GST compliance score: {v:.2f}/1.0 — {'regular filer' if v > 0.8 else 'irregular filings detected' if v > 0.5 else 'non-compliant, high risk'}.",
    'emi_delay_count':        lambda v: f"EMI delays recorded: {int(v)} — {'no delays' if v == 0 else 'payment stress visible'}.",
    'cashflow_stress_ratio':  lambda v: f"Cashflow stress index: {v:.2f} — {'severe' if v > 2.0 else 'moderate' if v > 1.0 else 'low'}.",
    'working_capital_usage':  lambda v: f"Working capital drawn: {v*100:.0f}% of sanctioned limit.",
    'revenue_trend_index':    lambda v: f"Revenue trend index: {v:.2f} — {'growing' if v > 1.1 else 'stable' if v > 0.9 else 'declining'}.",
    'payment_history_score':  lambda v: f"Payment history score: {v:.2f}/1.0 — {'excellent' if v > 0.9 else 'fair' if v > 0.7 else 'poor track record'}.",
    'supplier_payment_risk':  lambda v: f"Supplier payment risk flag: {'Active — delays to creditors detected' if v > 0 else 'Clear — no creditor delays'}.",
    'note_stress_index':      lambda v: f"Officer-notes stress index (NLP): {v:.2f}/1.0 — {'alarming language in notes' if v > 0.66 else 'some concern in notes' if v > 0.4 else 'notes read healthy'}.",
}

def get_risk_band(pd_value: float) -> str:
    if pd_value < 0.20: return "Low"
    if pd_value < 0.50: return "Medium"
    if pd_value < 0.75: return "High"
    return "Critical"

SECTORS    = ['Retail', 'Manufacturing', 'Construction', 'Technology', 'Agriculture', 'Textile']
LOAN_TYPES = ['Term Loan', 'Working Capital', 'Trade Credit']
SECTOR_RISK_BIAS = {'Retail': 0.1, 'Manufacturing': 0.15, 'Construction': 0.25,
                    'Technology': 0.08, 'Agriculture': 0.18, 'Textile': 0.20}


_STRESSED_NOTES = [
    "Severe cashflow pressure this quarter; EMI repeatedly delayed and supplier dues mounting.",
    "Revenue declining for months; borrower struggling to meet obligations, recovery concerns.",
    "Account inflows weak and overdrafts frequent; serious liquidity stress observed.",
    "Multiple missed payments and falling GST turnover; business health deteriorating rapidly.",
]
_NEUTRAL_NOTES = [
    "Operations broadly normal this period with minor fluctuations in sales.",
    "Some seasonal dip in turnover but repayments largely on schedule.",
    "Cashflow adequate; one delayed payment noted but otherwise stable.",
]
_HEALTHY_NOTES = [
    "Strong sales growth and healthy margins; all EMIs paid on time, GST filed promptly.",
    "Comfortable liquidity and rising deposits; well-managed, low-risk borrower.",
    "Stable, profitable operations; no concerns, excellent payment discipline.",
]

def _pick_note(rng: np.random.RandomState, risk_proxy: float) -> str:
    """Choose an officer note whose tone reflects the borrower's risk (with noise)."""
    r = rng.random()
    if risk_proxy >= 0.66:
        pool = _STRESSED_NOTES if r < 0.7 else _NEUTRAL_NOTES
    elif risk_proxy >= 0.33:
        pool = _NEUTRAL_NOTES if r < 0.6 else (_STRESSED_NOTES if r < 0.8 else _HEALTHY_NOTES)
    else:
        pool = _HEALTHY_NOTES if r < 0.7 else _NEUTRAL_NOTES
    return pool[rng.randint(len(pool))]


def _make_borrower_row(company_id: str, rng: np.random.RandomState, year: int = 2024) -> dict:
    """Generate one realistic MSME borrower in the model's feature space using `rng`."""
    sector = rng.choice(SECTORS)
    loan_type = rng.choice(LOAN_TYPES)
    bias = SECTOR_RISK_BIAS[sector]

    rev_util      = float(np.clip(rng.beta(2, 5) + bias * 0.5, 0.01, 0.99))
    debt_ratio    = float(np.clip(rng.exponential(0.35) + bias, 0.01, 3.0))
    late_3059     = int(rng.poisson(bias * 3))
    late_6089     = int(rng.poisson(bias * 1.5))
    late_90       = int(rng.poisson(bias * 0.8))
    open_lines    = int(rng.poisson(8) + 2)
    re_loans      = int(rng.poisson(1))
    num_dep       = int(rng.poisson(1.5))
    income        = float(np.clip(rng.lognormal(11, 0.8), 20000, 500000))

    gst_score     = float(np.clip(1 - late_3059 * 0.12 + rng.normal(0, 0.05), 0.0, 1.0))
    emi_delays    = min(late_3059 + late_6089, 12)
    cf_stress     = float(np.clip(debt_ratio * rng.uniform(0.8, 1.2), 0, 5))
    wc_usage      = float(np.clip(rev_util * 0.6 + rng.normal(0.1, 0.05), 0.0, 1.0))
    rev_trend     = float(np.clip(1.2 - debt_ratio * 0.4 + rng.normal(0, 0.1), 0.2, 2.0))
    pay_hist      = float(np.clip(1 - late_90 * 0.2 - late_3059 * 0.05, 0.0, 1.0))
    supp_risk     = float((late_3059 > 2) + (late_90 > 0))
    inc_stability = float(np.clip(income / 100000, 0.0, 1.0))
    outstanding   = float(rng.uniform(200000, 2500000))

    journey_events = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m_idx, month in enumerate(months):
        gst_ok = rng.random() > (bias * 0.6 + m_idx * 0.02)
        emi_ok = rng.random() > (bias * 0.5 + m_idx * 0.03)
        if not gst_ok:
            journey_events.append({"date": f"{year}-{m_idx+1:02d}-07", "type": "warning",
                                    "desc": f"GST filing delayed — {month} {year}"})
        if not emi_ok:
            journey_events.append({"date": f"{year}-{m_idx+1:02d}-15", "type": "alert",
                                    "desc": f"EMI payment overdue — {month} {year}"})
        elif m_idx == 0:
            journey_events.append({"date": f"{year}-{m_idx+1:02d}-15", "type": "ok",
                                    "desc": f"EMI paid on time — {month} {year}"})

    # Risk proxy → pick an officer note (unstructured signal); note_stress_index is
    # scored in batch by the caller (build_portfolio / generate_new_loans).
    risk_proxy = float(np.clip(
        0.35 * (rev_util) + 0.30 * min(emi_delays / 4, 1) + 0.25 * min(late_90, 2) / 2
        + 0.10 * (1 - pay_hist), 0, 1))
    officer_note = _pick_note(rng, risk_proxy)

    return {
        'company_id': company_id, 'sector': sector, 'loan_type': loan_type,
        'outstanding_loan': outstanding, 'officer_notes': officer_note,
        'revolving_utilization': rev_util, 'debt_ratio': debt_ratio,
        'late_30_59': late_3059, 'late_60_89': late_6089, 'late_90_days': late_90,
        'open_credit_lines': open_lines, 'real_estate_loans': re_loans,
        'num_dependents': num_dep, 'income_stability': inc_stability,
        'gst_compliance_score': gst_score, 'emi_delay_count': emi_delays,
        'cashflow_stress_ratio': cf_stress, 'working_capital_usage': wc_usage,
        'revenue_trend_index': rev_trend, 'payment_history_score': pay_hist,
        'supplier_payment_risk': supp_risk,
        'journey_events': json.dumps(journey_events),
    }


def build_portfolio() -> pd.DataFrame:
    """Build the deterministic 200-borrower demo portfolio."""
    rng = np.random.RandomState(2024)
    rows = [_make_borrower_row(f"MSME-{i+1:04d}", rng) for i in range(200)]
    df = pd.DataFrame(rows)
    df['note_stress_index'] = nlp_features.stress_index(df['officer_notes'])
    return df


# Monotonic counter so freshly-disbursed loans always get unique IDs across refreshes.
_new_loan_seq = 0

def generate_new_loans(count: int) -> pd.DataFrame:
    """Simulate `count` companies taking new loans (non-deterministic each call)."""
    global _new_loan_seq
    rng = np.random.RandomState()  # fresh entropy → genuinely new borrowers each refresh
    rows = []
    for _ in range(count):
        _new_loan_seq += 1
        rows.append(_make_borrower_row(f"MSME-N{_new_loan_seq:04d}", rng, year=2025))
    df = pd.DataFrame(rows)
    df['note_stress_index'] = nlp_features.stress_index(df['officer_notes'])
    return df

# ─── Upload: map an arbitrary CSV into the model's feature space ──
# Accepts two shapes:
#   (a) "model schema"  — rows already carry the 16 model FEATURES
#   (b) "blueprint MSME schema" — monthly rows with columns like
#       company_id, month, sector, outstanding_loan, monthly_sales,
#       gst_turnover, monthly_inflow, monthly_outflow, emi_delay_count,
#       credit_utilization, working_capital_usage, account_balance, default
#   Monthly data (a `month` column) is aggregated to the latest snapshot
#   per company and derived into the behavioural feature space.
DEFAULT_SECTOR = 'Retail'

def map_upload_to_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Collapse monthly journeys → one row per company (latest month)
    if 'month' in df.columns and 'company_id' in df.columns:
        df = _aggregate_monthly(df)

    # Meta columns
    if 'company_id' not in df.columns:
        df['company_id'] = [f"MSME-U{i+1:04d}" for i in range(len(df))]
    if 'sector' not in df.columns:
        df['sector'] = DEFAULT_SECTOR
    if 'loan_type' not in df.columns:
        df['loan_type'] = 'Term Loan'
    if 'outstanding_loan' not in df.columns:
        df['outstanding_loan'] = 1_000_000.0

    # If the file already carries the model features, keep them; otherwise derive.
    have = set(df.columns)
    missing = [f for f in FEATURES if (f not in have and f != 'note_stress_index')]
    if missing:
        df = _derive_features(df)

    # Unstructured data: score officer notes → note_stress_index (neutral if absent)
    if 'note_stress_index' not in df.columns:
        if 'officer_notes' in df.columns:
            df['note_stress_index'] = nlp_features.stress_index(df['officer_notes'])
        else:
            df['note_stress_index'] = 0.5

    # Final guarantee: every model feature exists and is numeric
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0
        df[f] = pd.to_numeric(df[f], errors='coerce')

    if 'journey_events' not in df.columns:
        df['journey_events'] = '[]'
    if 'officer_notes' not in df.columns:
        df['officer_notes'] = ''

    keep = ['company_id', 'sector', 'loan_type', 'outstanding_loan',
            'journey_events', 'officer_notes'] + FEATURES
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def _aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce blueprint monthly rows to one enriched snapshot per company."""
    rows = []
    for cid, grp in df.groupby('company_id'):
        grp = grp.sort_values('month')
        first, last = grp.iloc[0], grp.iloc[-1]
        rec = last.to_dict()

        def safe(a, b):
            return float(a) / float(b) if b not in (0, None) and not pd.isna(b) else 0.0

        # Trends across the observed window
        if 'monthly_sales' in grp:
            rec['revenue_trend_index'] = np.clip(
                safe(last.get('monthly_sales', 1), first.get('monthly_sales', 1)), 0.2, 2.0)
        if {'monthly_inflow', 'monthly_outflow'}.issubset(grp.columns):
            net = (grp['monthly_inflow'] - grp['monthly_outflow'])
            rec['_cashflow_vol'] = float(net.std()) if len(net) > 1 else 0.0
            rec['cashflow_stress_ratio'] = np.clip(
                safe(last['monthly_outflow'], last['monthly_inflow']) * 2.0, 0, 5)
        rows.append(rec)
    return pd.DataFrame(rows)


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map blueprint MSME columns → the 16 model features (mirrors training overlay)."""
    n = len(df)
    g = df.get

    util = pd.to_numeric(g('credit_utilization', pd.Series([0.4] * n)), errors='coerce').fillna(0.4)
    wc   = pd.to_numeric(g('working_capital_usage', pd.Series([0.5] * n)), errors='coerce').fillna(0.5)
    emi  = pd.to_numeric(g('emi_delay_count', pd.Series([0] * n)), errors='coerce').fillna(0)

    # Cashflow / debt proxies
    if {'monthly_inflow', 'monthly_outflow'}.issubset(df.columns):
        inflow  = pd.to_numeric(df['monthly_inflow'], errors='coerce').replace(0, np.nan)
        outflow = pd.to_numeric(df['monthly_outflow'], errors='coerce')
        debt = np.clip((outflow / inflow).fillna(0.5), 0.01, 3.0)
    else:
        debt = pd.Series(np.clip(util * 1.2, 0.01, 3.0), index=df.index)

    out = df.copy()
    out['revolving_utilization'] = np.clip(util, 0.01, 0.99)
    out['debt_ratio']            = debt
    out['late_30_59']            = np.clip(emi, 0, 12).astype(int)
    out['late_60_89']            = np.clip(emi - 1, 0, 12).astype(int)
    out['late_90_days']          = np.clip(emi - 2, 0, 12).astype(int)
    out['open_credit_lines']     = pd.to_numeric(g('open_credit_lines', pd.Series([8] * n)), errors='coerce').fillna(8)
    out['real_estate_loans']     = pd.to_numeric(g('real_estate_loans', pd.Series([1] * n)), errors='coerce').fillna(1)
    out['num_dependents']        = pd.to_numeric(g('num_dependents', pd.Series([1] * n)), errors='coerce').fillna(1)
    if 'account_balance' in df.columns:
        bal = pd.to_numeric(df['account_balance'], errors='coerce').fillna(0)
        out['income_stability'] = np.clip(bal / 500000, 0.0, 1.0)
    else:
        out['income_stability'] = 0.5
    out['gst_compliance_score']  = np.clip(1 - emi * 0.12, 0.0, 1.0)
    out['emi_delay_count']       = np.clip(emi, 0, 12).astype(int)
    if 'cashflow_stress_ratio' not in out.columns:
        out['cashflow_stress_ratio'] = np.clip(debt * 1.0, 0, 5)
    out['working_capital_usage'] = np.clip(wc, 0.0, 1.0)
    if 'revenue_trend_index' not in out.columns:
        out['revenue_trend_index'] = np.clip(1.2 - debt * 0.4, 0.2, 2.0)
    out['payment_history_score'] = np.clip(1 - emi * 0.08, 0.0, 1.0)
    out['supplier_payment_risk'] = (emi > 2).astype(float)
    return out


@app.on_event("startup")
def load_assets():
    global xgb_model, explainer, imputer, feature_df, FEATURES
    try:
        xgb_model = joblib.load('models/xgb_model.joblib')
        explainer = joblib.load('models/shap_explainer.joblib')
        imputer   = joblib.load('models/imputer.joblib')
        FEATURES  = joblib.load('models/feature_list.joblib')
        feature_df = build_portfolio()
        print(f"Real model loaded. Portfolio: {len(feature_df)} borrowers.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        raise

def predict_portfolio(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].copy()
    X_imp = imputer.transform(X)
    return xgb_model.predict_proba(X_imp)[:, 1]

# ─── API Routes ───────────────────────────────────────────────

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard/index.html")

@app.get("/model-performance")
def get_model_performance():
    """Return training performance metrics for dashboard display."""
    try:
        with open('models/performance_report.json') as f:
            return json.load(f)
    except:
        return {}

@app.get("/portfolio")
def get_portfolio_summary():
    if feature_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    probs = predict_portfolio(feature_df)
    df = feature_df.copy()
    df['pd']    = probs
    df['band']  = df['pd'].apply(get_risk_band)
    df['lgd']   = df['sector'].map(lambda s: LGD_MAP.get(s, 0.4))
    df['el']    = df['outstanding_loan'] * df['pd'] * df['lgd']

    sector_stats = []
    for sector, grp in df.groupby('sector'):
        sector_stats.append({
            "sector":         sector,
            "exposure":       float(grp['outstanding_loan'].sum()),
            "avg_pd":         float(grp['pd'].mean()),
            "expected_loss":  float(grp['el'].sum()),
            "borrower_count": int(len(grp))
        })

    dist = df['band'].value_counts().to_dict()
    for band in ["Low", "Medium", "High", "Critical"]:
        dist.setdefault(band, 0)

    return {
        "total_borrowers":    int(len(df)),
        "avg_pd":             float(probs.mean()),
        "total_exposure":     float(df['outstanding_loan'].sum()),
        "total_expected_loss": float(df['el'].sum()),
        "risk_distribution":  dist,
        "sector_analytics":   sorted(sector_stats, key=lambda x: x['expected_loss'], reverse=True)
    }

@app.get("/borrowers")
def get_borrowers(limit: int = 50):
    if feature_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    probs = predict_portfolio(feature_df)
    df = feature_df.copy()
    df['pd'] = probs
    top = df.sort_values('pd', ascending=False).head(limit)

    return [
        {
            "company_id":      row['company_id'],
            "sector":          row['sector'],
            "loan_type":       row['loan_type'],
            "pd":              float(row['pd']),
            "outstanding_loan": float(row['outstanding_loan']),
            "risk_band":       get_risk_band(float(row['pd']))
        }
        for _, row in top.iterrows()
    ]

@app.get("/borrowers/{company_id}")
def get_borrower_details(company_id: str):
    if feature_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Borrower not found")
    row = row.iloc[0]

    # Predict PD
    X = pd.DataFrame([row[FEATURES].to_dict()])
    X_imp = imputer.transform(X)
    current_pd = float(xgb_model.predict_proba(X_imp)[0, 1])

    # Build model-derived 12-month trajectory
    # Simulate degrading monthly features and run model at each step
    timeline = []
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    base_features = row[FEATURES].to_dict()
    bias = current_pd  # Use final PD as a proxy for underlying risk level

    for m_idx, month in enumerate(months):
        # Simulate monthly deterioration proportional to final PD
        scale = (m_idx / 11)  # 0 → 1
        monthly_feats = base_features.copy()
        monthly_feats['revolving_utilization'] = float(np.clip(
            base_features['revolving_utilization'] * (0.5 + 0.5 * scale), 0.01, 0.99))
        monthly_feats['gst_compliance_score'] = float(np.clip(
            base_features['gst_compliance_score'] * (1.1 - 0.2 * scale), 0, 1))
        monthly_feats['payment_history_score'] = float(np.clip(
            base_features['payment_history_score'] * (1.05 - 0.15 * scale), 0, 1))
        monthly_feats['cashflow_stress_ratio'] = float(
            base_features['cashflow_stress_ratio'] * (0.6 + 0.6 * scale))

        Xm = pd.DataFrame([monthly_feats])
        Xm_imp = imputer.transform(Xm)
        m_pd = float(xgb_model.predict_proba(Xm_imp)[0, 1])
        timeline.append({"month": month, "pd": m_pd})

    # Journey events
    journey_events = []
    try:
        raw = json.loads(row['journey_events'])
        seen = set()
        for ev in raw:
            key = ev['date'] + ev['desc']
            if key not in seen:
                seen.add(key)
                journey_events.append(ev)
    except:
        pass

    sector  = row['sector']
    lgd     = LGD_MAP.get(sector, 0.4)
    ead     = float(row['outstanding_loan'])
    el      = current_pd * ead * lgd

    return {
        "company_id":       company_id,
        "sector":           sector,
        "loan_type":        row.get('loan_type', 'Term Loan'),
        "outstanding_loan": ead,
        "lgd_pct":          lgd,
        "expected_loss":    el,
        "potential_recovery": ead - el,
        "current_pd":       current_pd,
        "risk_band":        get_risk_band(current_pd),
        "timeline":         timeline,
        "risk_migration": {
            "start_band": get_risk_band(timeline[0]['pd']),
            "end_band":   get_risk_band(current_pd)
        },
        "journey_events":  journey_events,
        "officer_notes":   str(row.get('officer_notes', '') or ''),
        "note_stress_index": float(row['note_stress_index']) if 'note_stress_index' in row else None,
        "raw_features":    {k: float(row[k]) for k in FEATURES},
        "action_ladder":   ACTION_LADDER.get(get_risk_band(current_pd), [])
    }

@app.get("/borrowers/{company_id}/explain")
def get_shap_explanation(company_id: str):
    if feature_df is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Borrower not found")

    X = row[FEATURES].copy()
    X_imp = imputer.transform(X)
    shap_vals = explainer.shap_values(X_imp)

    feat_vals = X.iloc[0].to_dict()
    drivers = sorted(
        [
            {
                "feature":   f,
                "value":     float(feat_vals[f]),
                "impact":    float(shap_vals[0][i]),
                "narrative": FEATURE_NARRATIVE.get(f, lambda x: f"Feature value: {x:.3f}")(float(feat_vals[f]))
            }
            for i, f in enumerate(FEATURES)
        ],
        key=lambda x: abs(x['impact']),
        reverse=True
    )
    return {"key_drivers": drivers[:8]}

# ─── Data Upload Module ──────────────────────────────────────

@app.post("/upload")
async def upload_portfolio(file: UploadFile = File(...)):
    """Upload a borrower CSV (model schema OR blueprint MSME schema).
    Replaces the active portfolio and re-scores it through the model."""
    global feature_df
    if FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    try:
        raw = await file.read()
        df_in = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    if df_in.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        mapped = map_upload_to_portfolio(df_in)
        # Validate it scores cleanly before committing
        _ = predict_portfolio(mapped)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not map file to model features: {e}")

    feature_df = mapped
    probs = predict_portfolio(feature_df)
    return {
        "status": "ok",
        "filename": file.filename,
        "rows_ingested": int(len(df_in)),
        "borrowers_scored": int(len(feature_df)),
        "avg_pd": float(probs.mean()),
        "high_risk_count": int((probs >= 0.5).sum()),
    }


@app.post("/reset")
def reset_portfolio():
    """Restore the built-in demo portfolio."""
    global feature_df
    feature_df = build_portfolio()
    return {"status": "ok", "borrowers": int(len(feature_df))}


@app.post("/refresh")
def refresh_portfolio(count: int = 10):
    """Simulate new companies taking loans and fold them into the live portfolio."""
    global feature_df
    if feature_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    count = max(1, min(int(count), 100))

    new_df = generate_new_loans(count)
    # Align columns with the active portfolio, then append
    feature_df = pd.concat([feature_df, new_df], ignore_index=True)

    probs = predict_portfolio(new_df)
    new_loans = sorted(
        [{"company_id": r['company_id'], "sector": r['sector'],
          "pd": float(p), "risk_band": get_risk_band(float(p)),
          "outstanding_loan": float(r['outstanding_loan'])}
         for (_, r), p in zip(new_df.iterrows(), probs)],
        key=lambda x: x['pd'], reverse=True)
    return {
        "status": "ok",
        "added": count,
        "total_borrowers": int(len(feature_df)),
        "new_high_risk": int((probs >= 0.5).sum()),
        "new_avg_pd": float(probs.mean()),
        "new_loans": new_loans,
    }

# ─── Agentic Risk Copilot (interactive Q&A) ──────────────────

def _borrower_context(company_id: str) -> dict:
    """Assemble PD, band, top SHAP drivers and financials for one borrower."""
    row = feature_df[feature_df['company_id'] == company_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Borrower not found")
    X = row[FEATURES].copy()
    X_imp = imputer.transform(X)
    pd_val = float(xgb_model.predict_proba(X_imp)[0, 1])
    band = get_risk_band(pd_val)

    drivers = []
    if explainer is not None:
        shap_vals = explainer.shap_values(X_imp)
        feat_vals = X.iloc[0].to_dict()
        drivers = sorted(
            [{"feature": f, "impact": float(shap_vals[0][i]),
              "narrative": FEATURE_NARRATIVE.get(f, lambda x: f"{f}: {x:.3f}")(float(feat_vals[f]))}
             for i, f in enumerate(FEATURES)],
            key=lambda d: abs(d['impact']), reverse=True)[:5]

    r = row.iloc[0]
    sector = r['sector']
    lgd = LGD_MAP.get(sector, 0.4)
    ead = float(r['outstanding_loan'])
    return {
        "company_id": company_id, "sector": sector, "pd": pd_val, "band": band,
        "ead": ead, "lgd": lgd, "expected_loss": pd_val * ead * lgd,
        "drivers": drivers,
        "officer_notes": str(r.get('officer_notes', '') or ''),
        "note_stress_index": float(r['note_stress_index']) if 'note_stress_index' in r else None,
        "actions": [a["action"] for a in ACTION_LADDER.get(band, [])],
    }


def _portfolio_attention() -> list:
    probs = predict_portfolio(feature_df)
    df = feature_df.copy()
    df['pd'] = probs
    top = df.sort_values('pd', ascending=False).head(5)
    return [{"company_id": r['company_id'], "sector": r['sector'],
             "pd": float(r['pd']), "band": get_risk_band(float(r['pd']))}
            for _, r in top.iterrows()]


def copilot_engine() -> str:
    """Pick the Copilot backend by which API key is present.
    Priority: Claude → Gemini → offline templates."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "template"


COPILOT_SYSTEM = (
    "You are a credit risk officer's copilot at a bank. Answer the manager's question "
    "using ONLY the structured borrower/portfolio context provided. Be concise, factual, "
    "and write in plain manager-friendly English. Do not invent numbers."
)


def _copilot_context_json(ctx: dict | None) -> str:
    if ctx is None:
        return json.dumps({"portfolio_attention": _portfolio_attention()}, indent=2)
    return json.dumps(ctx, indent=2)


def _answer_template(question: str, ctx: dict | None) -> str:
    q = question.lower()
    if ctx is None:  # portfolio-level
        rows = _portfolio_attention()
        lines = [f"• {r['company_id']} ({r['sector']}) — PD {r['pd']*100:.1f}%, {r['band']}" for r in rows]
        return "Borrowers requiring immediate attention (highest PD):\n" + "\n".join(lines)

    drivers = "; ".join(d['narrative'] for d in ctx['drivers'][:3]) or "no dominant signals"
    if "action" in q or "do" in q or "recommend" in q:
        acts = ", ".join(ctx['actions']) or "Continue standard monitoring"
        return (f"{ctx['company_id']} is in the {ctx['band']} band (PD {ctx['pd']*100:.1f}%). "
                f"Recommended interventions: {acts}.")
    if "changed" in q or "trend" in q or "month" in q or "six" in q:
        return (f"Over the observation window the dominant deteriorating signals for {ctx['company_id']} are: "
                f"{drivers}. This drove the PD to {ctx['pd']*100:.1f}% ({ctx['band']} risk).")
    # default: why risky
    return (f"{ctx['company_id']} ({ctx['sector']}) carries a {ctx['pd']*100:.1f}% probability of default "
            f"({ctx['band']} band). Key drivers: {drivers}. Projected expected loss is "
            f"₹{ctx['expected_loss']:,.0f} (PD × EAD ₹{ctx['ead']:,.0f} × LGD {ctx['lgd']*100:.0f}%).")


def _answer_claude(question: str, ctx: dict | None) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
        system=COPILOT_SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context:\n{_copilot_context_json(ctx)}\n\nManager's question: {question}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


def _answer_gemini(question: str, ctx: dict | None) -> str:
    import time
    from google import genai
    from google.genai import types
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    config = types.GenerateContentConfig(
        system_instruction=COPILOT_SYSTEM,
        max_output_tokens=600,
        # Disable "thinking" so the token budget goes to the actual answer (and it's faster).
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    contents = f"Context:\n{_copilot_context_json(ctx)}\n\nManager's question: {question}"

    # Gemini's free endpoint intermittently throws transient 5xx / network errors;
    # retry those with backoff. Do NOT retry ClientError (e.g. 429 quota) — bubble up fast.
    TRANSIENT = ("ServerError", "ReadError", "ConnectError", "ReadTimeout",
                 "ConnectTimeout", "RemoteProtocolError", "APIError")
    last_err = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            text = (resp.text or "").strip()
            if text:
                return text
            last_err = RuntimeError("empty response")
        except genai.errors.ClientError:
            raise  # 4xx (quota/invalid) — retrying won't help quickly
        except Exception as e:
            if type(e).__name__ not in TRANSIENT:
                raise
            last_err = e
        time.sleep(1.2 * (attempt + 1))
    raise last_err


@app.post("/copilot")
def copilot(payload: dict = Body(...)):
    """Interactive Risk Copilot. Body: {question, company_id?}.
    Answers per-borrower, or portfolio-level if company_id is omitted."""
    if feature_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    company_id = payload.get("company_id")

    ctx = _borrower_context(company_id) if company_id else None
    engine = copilot_engine()
    try:
        if engine == "claude":
            answer = _answer_claude(question, ctx)
        elif engine == "gemini":
            answer = _answer_gemini(question, ctx)
        else:
            answer = _answer_template(question, ctx)
    except Exception as e:
        # Never fail the demo — fall back to the local template engine
        answer = _answer_template(question, ctx)
        engine = f"template (fallback: {type(e).__name__})"
    return {"engine": engine, "company_id": company_id, "question": question, "answer": answer}
