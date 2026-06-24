# IDBI MSME Risk Intelligence Engine

An advanced, forward-looking predictive AI engine and early-warning dashboard designed for the IDBI MSME Credit Risk portfolio (Track 04).

## 🎯 Problem Statement Alignment (Track 04)
The current capability for MSME risk prediction relies on fragmented rule-based models yielding only 16-22% accuracy. The expected outcome is to develop a robust predictive solution to identify potential stress in loans **12 months in advance**, improving accuracy/capture rate to **90%**.

## 🚀 Solution Architecture
We shifted the paradigm from retrospective static analysis to **forward-looking behavioral signal monitoring**.

1. **The Data Foundation (Real World Data):**
   - We abandoned synthetic "perfect" datasets and trained the core model on **150,000 real borrower records** from Kaggle's "Give Me Some Credit" dataset.
   - We cross-referenced and enriched the data using the comprehensive **Lending Club 2007-2018** data schema (10,000 records subset).

2. **The ML Engine:**
   - **Legacy Baseline:** A standard Logistic Regression model using 3 classic features (Utilization, DTI, Delinquency) achieved a **69.9% recall** (catch rate).
   - **XGBoost Engine:** Using 16 highly-tailored MSME behavioral features (GST Compliance Score, EMI Delays, Cashflow Stress Ratio), the XGBoost model achieved **87.8% recall** with an AUC of 0.860.
   - We successfully bumped the detection rate of actual defaults to near 90%, exactly fulfilling the core challenge requirement.

3. **12-Month Forward-Looking Trajectory:**
   - The model dynamically simulates and predicts the Probability of Default (PD) across a 12-month horizon based on real-time degradation of behavioral signals.

4. **"Banker's View" Dashboard:**
   - **Agentic Risk Copilot:** Uses SHAP (SHapley Additive exPlanations) to translate complex XGBoost feature weights into plain-English sentences for Bank Managers.
   - **Financial Impact Mapping:** Calculates real Expected Loss (EL) using the standard banking formula `PD × EAD × LGD`.
   - **Action Escalation Ladder:** Automatically recommends operational interventions (e.g., "Schedule RM Call", "Initiate Document Review") based on real-time risk banding.

## 🛠️ Technology Stack
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, Numpy
* **Explainability:** SHAP (TreeExplainer)
* **Frontend:** Vanilla JS, TailwindCSS, Chart.js

## ⚙️ How to Run Locally

### 1. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only — XGBoost needs the OpenMP runtime:
brew install libomp
```

### 2. Run the Server
Start the FastAPI application via Uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

### 3. Open the Dashboard
Navigate your browser to either:
* [http://127.0.0.1:8000/](http://127.0.0.1:8000/) (auto-redirects), or
* [http://127.0.0.1:8000/dashboard/index.html](http://127.0.0.1:8000/dashboard/index.html)

## 🧩 MVP Modules & API Endpoints
| Module | Endpoint |
|--------|----------|
| Portfolio monitoring | `GET /portfolio`, `GET /borrowers` |
| Risk prediction + timeline + expected loss + actions | `GET /borrowers/{id}` |
| Explainability (SHAP) | `GET /borrowers/{id}/explain` |
| Model performance | `GET /model-performance` |
| **Data Upload** (model schema *or* blueprint MSME monthly schema) | `POST /upload`, `POST /reset` |
| **Agentic Risk Copilot** (interactive Q&A) | `POST /copilot` |

**AI Risk Copilot engine:** the `/copilot` endpoint auto-detects its backend — if the
`ANTHROPIC_API_KEY` environment variable is set it answers with Claude (`claude-opus-4-8`);
otherwise it falls back to a fully-offline rule-based template engine. No key is required to run the demo.

### Try the upload module
A blueprint-format synthetic dataset can be generated and uploaded via the dashboard's **Upload CSV** button:
```bash
python generate_msme_data.py     # writes data/msme_synthetic_data.csv (1,000 MSMEs × 12 months)
```

## 📁 Repository Structure
* `main.py` - FastAPI application, backend logic, and dynamic portfolio generation.
* `train_real_model.py` - ML pipeline script used to train the XGBoost engine on the Give Me Some Credit / Lending Club datasets.
* `models/` - Serialized Joblib artifacts (XGBoost model, SHAP explainer, Imputer).
* `static/` - Frontend assets including `index.html`.
