import pandas as pd
import numpy as np
import random
import os
import json
from datetime import datetime, timedelta

def generate_msme_data(num_companies=1000):
    np.random.seed(42)
    random.seed(42)
    
    sectors = ['Manufacturing', 'Retail', 'Construction', 'Technology']
    
    records = []
    
    # Base date: Jan 2026
    start_date = datetime(2026, 1, 1)
    
    for i in range(1, num_companies + 1):
        company_id = f'C{str(i).zfill(5)}'
        sector = random.choice(sectors)
        
        outstanding_loan = np.random.uniform(500000, 5000000)
        will_default = np.random.rand() < 0.15
        
        base_sales = np.random.uniform(500000, 5000000)
        base_gst = base_sales * np.random.uniform(0.8, 1.0)
        base_inflow = base_sales * np.random.uniform(0.9, 1.1)
        base_outflow = base_sales * np.random.uniform(0.7, 0.9)
        base_balance = np.random.uniform(100000, 1000000)
        
        credit_utilization = np.random.uniform(0.3, 0.6)
        working_capital_usage = np.random.uniform(0.4, 0.7)
        emi_delays = 0
        
        journey_events = []
        
        for month in range(1, 13):
            noise_factor = np.random.normal(1.0, 0.1)
            
            # Event Date approx
            event_date = start_date + timedelta(days=(month-1)*30 + random.randint(1, 15))
            date_str = event_date.strftime('%Y-%m-%d')
            
            monthly_events = []
            
            if will_default:
                decline_factor = 1.0 - (month * 0.04 * np.random.uniform(0.5, 1.5))
                sales = base_sales * decline_factor * noise_factor
                gst = base_gst * decline_factor * noise_factor
                inflow = base_inflow * decline_factor * noise_factor
                outflow = base_outflow * np.random.uniform(0.9, 1.1)
                balance = base_balance - (outflow - inflow)
                if balance < 0: balance = 0
                
                credit_utilization = min(1.0, credit_utilization + (month * 0.03 * noise_factor))
                working_capital_usage = min(1.0, working_capital_usage + (month * 0.03 * noise_factor))
                
                # Behavioral Journey generation
                if month < 3:
                    monthly_events.append({"date": date_str, "type": "success", "desc": "GST Filed ✓"})
                    monthly_events.append({"date": date_str, "type": "success", "desc": "EMI Paid ✓"})
                else:
                    if np.random.rand() > 0.4:
                        delay_days = random.randint(5, 25)
                        monthly_events.append({"date": date_str, "type": "warning", "desc": f"GST Filed {delay_days} Days Late"})
                    
                    if np.random.rand() > 0.6:
                        emi_delays += 1
                        monthly_events.append({"date": date_str, "type": "critical", "desc": "EMI Missed"})
                        
                    if decline_factor < 0.8 and np.random.rand() > 0.5:
                        monthly_events.append({"date": date_str, "type": "warning", "desc": f"Monthly Revenue Down {random.randint(10, 30)}%"})
                        
                    if working_capital_usage > 0.85:
                        monthly_events.append({"date": date_str, "type": "critical", "desc": "Supplier Payment Delay Detected"})

                notes = "Cashflow pressure observed." if month > 6 else "Normal operations."
                
            else:
                # Stable company
                sales = base_sales * noise_factor
                gst = base_gst * noise_factor
                inflow = base_inflow * noise_factor
                outflow = base_outflow * noise_factor
                balance = base_balance + (inflow - outflow)
                
                credit_utilization = max(0.1, min(0.9, credit_utilization + np.random.uniform(-0.05, 0.05)))
                working_capital_usage = max(0.1, min(0.9, working_capital_usage + np.random.uniform(-0.05, 0.05)))
                
                monthly_events.append({"date": date_str, "type": "success", "desc": "GST Filed ✓"})
                
                if month > 6 and np.random.rand() > 0.95:
                    emi_delays += 1
                    monthly_events.append({"date": date_str, "type": "warning", "desc": "EMI Payment Delayed by 3 days"})
                else:
                    monthly_events.append({"date": date_str, "type": "success", "desc": "EMI Paid ✓"})
                
                notes = "Business running normally."
            
            # Add monthly events to the master journey
            journey_events.extend(monthly_events)

            records.append({
                'company_id': company_id,
                'month': month,
                'sector': sector,
                'outstanding_loan': round(outstanding_loan, 2),
                'monthly_sales': round(sales, 2),
                'gst_turnover': round(gst, 2),
                'monthly_inflow': round(inflow, 2),
                'monthly_outflow': round(outflow, 2),
                'emi_delay_count': emi_delays,
                'credit_utilization': round(credit_utilization, 4),
                'working_capital_usage': round(working_capital_usage, 4),
                'account_balance': round(balance, 2),
                'officer_notes': notes,
                'journey_events': json.dumps(journey_events), # Store cumulative journey up to this month
                'default': 1 if will_default else 0
            })
            
            if not will_default:
                base_balance = balance

    df = pd.DataFrame(records)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/msme_synthetic_data.csv', index=False)
    print(f"Generated {len(df)} records for {num_companies} companies with Borrower Journeys.")
    print("Saved to data/msme_synthetic_data.csv")

if __name__ == '__main__':
    generate_msme_data(1000)
