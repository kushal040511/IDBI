import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
import shap
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

def extract_nlp_features(notes_series):
    print("Extracting NLP features from unstructured officer notes...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(notes_series.tolist(), show_progress_bar=True)
    
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    return reduced_embeddings[:, 0], reduced_embeddings[:, 1]

def engineer_features(df):
    companies = df.groupby('company_id')
    features = []
    
    for name, group in companies:
        group = group.sort_values('month')
        m1 = group.iloc[0]
        
        # Calculate features at month 6, 7, 8, 9, 10, 11, 12 to create a timeline of features
        for t in range(6, 13):
            sub_group = group[group['month'] <= t]
            current = sub_group.iloc[-1]
            
            gst_decline_pct = (m1['gst_turnover'] - current['gst_turnover']) / (m1['gst_turnover'] + 1e-5)
            revenue_decline_pct = (m1['monthly_sales'] - current['monthly_sales']) / (m1['monthly_sales'] + 1e-5)
            balance_decline_pct = (m1['account_balance'] - current['account_balance']) / (m1['account_balance'] + 1e-5)
            
            total_emi_delays = sub_group['emi_delay_count'].sum()
            utilization_growth = current['credit_utilization'] - m1['credit_utilization']
            
            cashflows = sub_group['monthly_inflow'] - sub_group['monthly_outflow']
            cashflow_volatility = cashflows.std() if len(cashflows) > 1 else 0
            avg_emi_to_rev = (sub_group['monthly_outflow'] * 0.2) / (sub_group['monthly_sales'] + 1e-5)
            
            features.append({
                'company_id': name,
                'month': t,
                'sector': current['sector'],
                'outstanding_loan': current['outstanding_loan'],
                'gst_decline_pct': gst_decline_pct,
                'revenue_decline_pct': revenue_decline_pct,
                'balance_decline_pct': balance_decline_pct,
                'total_emi_delays': total_emi_delays,
                'utilization_growth': utilization_growth,
                'cashflow_volatility': cashflow_volatility,
                'avg_emi_to_revenue': avg_emi_to_rev.mean(),
                'officer_note': current['officer_notes'],
                'journey_events': current['journey_events'],
                'default': current['default'] # Target is whether they ultimately default
            })
        
    feature_df = pd.DataFrame(features)
    
    nlp_f1, nlp_f2 = extract_nlp_features(feature_df['officer_note'])
    feature_df['nlp_stress_flag_1'] = nlp_f1
    feature_df['nlp_stress_flag_2'] = nlp_f2
    
    feature_df = feature_df.drop(columns=['officer_note'])
    return feature_df

def train_and_evaluate():
    print("Loading data...")
    df = pd.read_csv('data/msme_synthetic_data.csv')
    
    print("Engineering features (creating monthly snapshots)...")
    feature_df = engineer_features(df)
    
    # Train the model only on the final state (Month 12) so it learns the final default boundary
    train_df = feature_df[feature_df['month'] == 12]
    
    X = train_df.drop(columns=['company_id', 'month', 'sector', 'outstanding_loan', 'journey_events', 'default'])
    y = train_df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Main Engine: XGBoost (Calibrated) ---")
    # Add strong regularization to prevent extreme 99.2% clustering
    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42,
        max_depth=3,
        learning_rate=0.05,
        reg_lambda=10,
        min_child_weight=5,
        n_estimators=100
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
    print(f"Precision: {precision_score(y_test, xgb_preds):.4f}")
    print(f"Recall: {recall_score(y_test, xgb_preds):.4f}")
    
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(xgb_model)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_model.joblib')
    joblib.dump(explainer, 'models/shap_explainer.joblib')
    
    # Save the full historical features for the backend to run the timeline
    feature_df.to_csv('data/processed_features.csv', index=False)
    print("Models and processed historical data saved.")

if __name__ == '__main__':
    train_and_evaluate()
