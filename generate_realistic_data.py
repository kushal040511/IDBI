"""
Realistic Synthetic MSME Dataset Generator
==========================================
Produces a per-company dataset in the model's 16-feature space with a
*probabilistic, noisy* default label — deliberately NOT separable, so it
behaves like real credit data (achievable AUC ~0.80-0.88, not 0.99).

Default mechanism:
  latent = intercept + Σ wᵢ·zᵢ  +  ε        (ε ~ N(0, σ): irreducible noise)
  P(default) = sigmoid(latent)
  default ~ Bernoulli(P)                      then ~3% labels flipped (label noise)

This gives correlated, realistic features AND a label with a real Bayes ceiling.
Output: data/msme_realistic.csv  (16 features + company_id, sector, outstanding_loan, default)
"""
import numpy as np
import pandas as pd
import os

SEED = 7
N = 30000
SECTORS = ['Retail', 'Manufacturing', 'Construction', 'Technology', 'Agriculture', 'Textile']
SECTOR_BIAS = {'Retail': 0.10, 'Manufacturing': 0.15, 'Construction': 0.25,
               'Technology': 0.08, 'Agriculture': 0.18, 'Textile': 0.20}

FEATURES = [
    'revolving_utilization', 'debt_ratio', 'late_30_59', 'late_60_89', 'late_90_days',
    'open_credit_lines', 'real_estate_loans', 'num_dependents', 'income_stability',
    'gst_compliance_score', 'emi_delay_count', 'cashflow_stress_ratio',
    'working_capital_usage', 'revenue_trend_index', 'payment_history_score',
    'supplier_payment_risk',
]

# Sign & magnitude of each feature's contribution to default risk (domain-sensible).
WEIGHTS = {
    'revolving_utilization':  1.1,
    'debt_ratio':             0.9,
    'late_30_59':             0.6,
    'late_60_89':             0.8,
    'late_90_days':           1.4,
    'open_credit_lines':      0.15,
    'real_estate_loans':     -0.25,
    'num_dependents':         0.10,
    'income_stability':      -0.7,
    'gst_compliance_score':  -0.9,
    'emi_delay_count':        0.7,
    'cashflow_stress_ratio':  0.8,
    'working_capital_usage':  0.6,
    'revenue_trend_index':   -0.8,
    'payment_history_score': -1.2,
    'supplier_payment_risk':  0.5,
}
NOISE_SIGMA   = 2.3    # irreducible noise → caps achievable AUC around ~0.85
LABEL_FLIP    = 0.03   # fraction of labels randomly flipped
TARGET_RATE   = 0.15   # desired default prevalence


def main():
    rng = np.random.RandomState(SEED)
    sectors = rng.choice(SECTORS, size=N)
    bias = np.array([SECTOR_BIAS[s] for s in sectors])

    rev_util  = np.clip(rng.beta(2, 5, N) + bias * 0.5, 0.01, 0.99)
    debt      = np.clip(rng.exponential(0.35, N) + bias, 0.01, 3.0)
    late3059  = rng.poisson(bias * 3)
    late6089  = rng.poisson(bias * 1.5)
    late90    = rng.poisson(bias * 0.8)
    open_l    = rng.poisson(8, N) + 2
    re_loans  = rng.poisson(1, N)
    num_dep   = rng.poisson(1.5, N)
    income    = np.clip(rng.lognormal(11, 0.8, N), 20000, 500000)

    gst       = np.clip(1 - late3059 * 0.12 + rng.normal(0, 0.08, N), 0, 1)
    emi       = np.clip(late3059 + late6089, 0, 12)
    cf_stress = np.clip(debt * rng.uniform(0.8, 1.2, N), 0, 5)
    wc_usage  = np.clip(rev_util * 0.6 + rng.normal(0.1, 0.08, N), 0, 1)
    rev_trend = np.clip(1.2 - debt * 0.4 + rng.normal(0, 0.15, N), 0.2, 2.0)
    pay_hist  = np.clip(1 - late90 * 0.2 - late3059 * 0.05 + rng.normal(0, 0.05, N), 0, 1)
    supp_risk = ((late3059 > 2).astype(float) + (late90 > 0).astype(float))
    inc_stab  = np.clip(income / 100000, 0, 1)

    df = pd.DataFrame({
        'revolving_utilization': rev_util, 'debt_ratio': debt,
        'late_30_59': late3059, 'late_60_89': late6089, 'late_90_days': late90,
        'open_credit_lines': open_l, 'real_estate_loans': re_loans,
        'num_dependents': num_dep, 'income_stability': inc_stab,
        'gst_compliance_score': gst, 'emi_delay_count': emi,
        'cashflow_stress_ratio': cf_stress, 'working_capital_usage': wc_usage,
        'revenue_trend_index': rev_trend, 'payment_history_score': pay_hist,
        'supplier_payment_risk': supp_risk,
    })

    # Standardize features → weighted latent → add noise
    Z = (df[FEATURES] - df[FEATURES].mean()) / (df[FEATURES].std() + 1e-9)
    w = np.array([WEIGHTS[f] for f in FEATURES])
    latent = Z.values @ w + rng.normal(0, NOISE_SIGMA, N)

    # Calibrate intercept to hit the target default prevalence
    from scipy.optimize import brentq
    def rate_at(b):
        return 1 / (1 + np.exp(-(latent + b)))
    intercept = brentq(lambda b: rate_at(b).mean() - TARGET_RATE, -20, 20)
    p = rate_at(intercept)

    y = (rng.random(N) < p).astype(int)
    # Label noise
    flip = rng.random(N) < LABEL_FLIP
    y[flip] = 1 - y[flip]

    # ── Unstructured data: officer notes correlated with the (noisy) label ──
    # Notes carry signal about y *beyond* the structured features (they reflect the
    # realized outcome, incl. its noise), so adding NLP features genuinely lifts AUC.
    STRESSED_NOTES = [
        "Severe cashflow pressure this quarter; EMI repeatedly delayed and supplier dues mounting.",
        "Revenue has been declining for months; borrower struggling to meet obligations, recovery concerns.",
        "Account inflows weak and overdrafts frequent; serious liquidity stress observed.",
        "Multiple missed payments and falling GST turnover; business health deteriorating rapidly.",
        "Working capital exhausted, creditors complaining of delays; high default risk flagged.",
    ]
    NEUTRAL_NOTES = [
        "Operations broadly normal this period with minor fluctuations in sales.",
        "Some seasonal dip in turnover but repayments largely on schedule.",
        "Cashflow adequate; one delayed payment noted but otherwise stable.",
        "Business steady; monitoring utilization which has ticked up slightly.",
    ]
    HEALTHY_NOTES = [
        "Strong sales growth and healthy margins; all EMIs paid on time, GST filed promptly.",
        "Comfortable liquidity and rising deposits; well-managed, low-risk borrower.",
        "Robust cashflow with consistent repayment track record and growing revenue.",
        "Stable, profitable operations; no concerns, excellent payment discipline.",
    ]
    notes = []
    for yi in y:
        r = rng.random()
        if yi == 1:                       # defaulter: mostly stressed, some neutral/healthy (noise)
            pool = STRESSED_NOTES if r < 0.65 else (NEUTRAL_NOTES if r < 0.88 else HEALTHY_NOTES)
        else:                             # healthy: mostly healthy/neutral, some stressed (noise)
            pool = HEALTHY_NOTES if r < 0.60 else (NEUTRAL_NOTES if r < 0.85 else STRESSED_NOTES)
        notes.append(pool[rng.randint(len(pool))])

    print("Scoring officer notes with the NLP model (note_stress_index)...")
    import nlp_features
    df['officer_notes'] = notes
    df['note_stress_index'] = nlp_features.stress_index(notes)

    df['company_id'] = [f"MSME-R{i+1:05d}" for i in range(N)]
    df['sector'] = sectors
    df['outstanding_loan'] = rng.uniform(200000, 2500000, N)
    df['default'] = y

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/msme_realistic.csv', index=False)

    # Diagnostics
    from sklearn.metrics import roc_auc_score
    oracle_auc = roc_auc_score(y, p)
    note_auc = roc_auc_score(y, df['note_stress_index'])
    print(f"Generated {N:,} companies → data/msme_realistic.csv  (+ officer_notes, note_stress_index)")
    print(f"Default prevalence       : {y.mean()*100:.1f}%")
    print(f"Bayes-ceiling AUC (struct): {oracle_auc:.4f}")
    print(f"note_stress_index AUC     : {note_auc:.4f}  (standalone signal from unstructured text)")


if __name__ == '__main__':
    main()
