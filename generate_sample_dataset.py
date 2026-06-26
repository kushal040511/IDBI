"""
Sample Portfolio Generator (for the Upload CSV demo)
====================================================
Creates a realistic, *named* MSME portfolio in the blueprint monthly schema with a
deliberate spread across all four risk bands (Low / Medium / High / Critical), so
uploading it to the dashboard exercises the full pipeline and looks compelling.

Output: data/sample_portfolio.csv   (one row per company per month, 12 months)
Upload it via the dashboard's "Upload CSV" button.
"""
import numpy as np
import pandas as pd
import json, os

rng = np.random.RandomState(101)

NAMES = {
    'Manufacturing': ['Shakti Steel Works', 'Pioneer Auto Components', 'Gokul Castings',
                      'Apex Precision Tools', 'Vimal Industrial Fabricators'],
    'Retail':        ['Sunrise Supermart', 'Anand General Stores', 'Metro Fashion Hub',
                      'Krishna Electronics', 'Daily Needs Mart'],
    'Construction':  ['Skyline Infra Builders', 'Konkan Cement Works', 'Sterling Realty Projects',
                      'Heritage Construction Co', 'Vajra Infrastructures'],
    'Technology':    ['ByteForge Solutions', 'NovaSoft Systems', 'CloudNine IT Services',
                      'Quantum Data Labs'],
    'Agriculture':   ['Greenfield Agro Farms', 'Annapurna Foods', 'Sahyadri Dairy Co-op',
                      'Bharat Seeds & Fertilizers'],
    'Textile':       ['Rajwadi Textiles', 'Coimbatore Spinning Mills', 'Silk Route Exports',
                      'Comfort Weaves Ltd'],
}

# risk tier → behavioural parameters (calibrated against the tuned model so each
# tier reliably lands in its intended band: Low / Medium / High / Critical)
TIERS = {
    'Low':      dict(util=(0.20, 0.33), out_in=(0.70, 0.82), decline=0.000, emi_total=0,        emi_months=[]),
    'Medium':   dict(util=(0.44, 0.50), out_in=(0.90, 0.95), decline=0.010, emi_total=1,        emi_months=[9]),
    'High':     dict(util=(0.54, 0.575),out_in=(0.96, 0.99), decline=0.020, emi_total=1,        emi_months=[8]),
    'Critical': dict(util=(0.80, 0.96), out_in=(1.12, 1.30), decline=0.070, emi_total=(5, 9),   emi_months='late'),
}
TIER_SEQ = ['Low', 'Medium', 'High', 'Critical']
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

rows = []
cid = 0
for sector, names in NAMES.items():
    for name in names:
        tier = TIER_SEQ[cid % 4]          # rotate tiers → even spread across bands
        cid += 1
        cfg = TIERS[tier]

        outstanding = float(rng.uniform(500_000, 5_000_000))
        base_sales  = float(rng.uniform(600_000, 4_000_000))
        base_bal    = float(rng.uniform(150_000, 1_200_000))
        util0       = float(rng.uniform(*cfg['util']))
        out_in      = float(rng.uniform(*cfg['out_in']))
        emi_target  = cfg['emi_total'] if isinstance(cfg['emi_total'], int) else int(rng.randint(*cfg['emi_total']))
        if cfg['emi_months'] == 'late':
            emi_set = set(rng.choice(range(3, 12), size=min(emi_target, 9), replace=False))
        else:
            emi_set = set(cfg['emi_months'][:emi_target])
        journey = []
        emi_delays = 0

        for m in range(12):
            sales   = base_sales * (1 - cfg['decline'] * m) * rng.normal(1.0, 0.04)
            gst     = sales * rng.uniform(0.84, 0.97)
            inflow  = sales * rng.uniform(0.92, 1.04)
            outflow = inflow * out_in
            base_bal = max(0.0, base_bal + (inflow - outflow))
            # utilisation drifts up only mildly; tight noise so the band stays put
            util = float(np.clip(util0 + cfg['decline'] * 0.3 * m / 11 + rng.normal(0, 0.008), 0.05, 0.99))
            wc   = float(np.clip(util + 0.04, 0.05, 0.99))

            date = f"2025-{m+1:02d}-12"
            if m in emi_set:
                emi_delays += 1
                journey.append({"date": date, "type": "critical", "desc": f"EMI Missed — {MONTHS[m]} 2025"})
                journey.append({"date": date, "type": "warning",
                                "desc": f"GST Filed {int(rng.randint(8,25))} Days Late — {MONTHS[m]} 2025"})
            else:
                journey.append({"date": date, "type": "success", "desc": f"EMI Paid ✓ — {MONTHS[m]} 2025"})
                journey.append({"date": date, "type": "success", "desc": f"GST Filed ✓ — {MONTHS[m]} 2025"})

            rows.append({
                'company_id': name,
                'month': m + 1,
                'sector': sector,
                'outstanding_loan': round(outstanding, 2),
                'monthly_sales': round(sales, 2),
                'gst_turnover': round(gst, 2),
                'monthly_inflow': round(inflow, 2),
                'monthly_outflow': round(outflow, 2),
                'emi_delay_count': emi_delays,
                'credit_utilization': round(util, 4),
                'working_capital_usage': round(wc, 4),
                'account_balance': round(base_bal, 2),
                'officer_notes': ("Cashflow pressure observed." if tier in ('High', 'Critical') and m > 5
                                  else "Operations normal."),
                'journey_events': json.dumps(journey),
            })

df = pd.DataFrame(rows)
os.makedirs('data', exist_ok=True)
out = 'data/sample_portfolio.csv'
df.to_csv(out, index=False)
print(f"Wrote {out}: {df['company_id'].nunique()} companies × 12 months = {len(df)} rows")
print("Upload it via the dashboard's 'Upload CSV' button.")
