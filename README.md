# 🚂 Railway Maintenance AI System

An end-to-end AI-powered maintenance scheduling system for **Indian Railways**, integrating four real-world railway subsystems with Machine Learning and Constraint-Based Optimization.

---

## 📐 System Architecture

```
data_synthesizer.py  →  Generates synthetic TMS / SMMS / TDMS / COA datasets
        ↓
ai_engine.py         →  Trains XGBoost (risk) + Random Forest (duration) models
        ↓
optimizer.py         →  OR-Tools CP-SAT produces optimized maintenance schedule
        ↓
api.py               →  FastAPI REST backend + Gemini LLM copilot
```

---

## 🏗️ Subsystem Definitions

| Code   | Full Name | Domain |
|--------|-----------|--------|
| **TMS**  | Track Management System | Track/Engineering defects — rail geometry, gauge, ultrasonic |
| **SMMS** | Signalling Maintenance & Management System | Signal & Telecom — point machines, track circuits, relays |
| **TDMS** | Traction Distribution Management System | Overhead electrical — 25kV OHE, contact wire, boosters |
| **COA**  | Control Office Application | Train timetables, goods forecasts, corridor block availability |

---

## 🤖 ML Models

### Model 1 — XGBoost Risk Classifier
- **Target**: `risk_label` (0 = low risk, 1 = high risk)
- **Key features**: asset age, fault count, inspection gap, wear %, traffic load
- **Class imbalance**: Handled via `scale_pos_weight` (no data discarded)
- **Tuning**: `GridSearchCV` with `StratifiedKFold(5)`
- **Metric**: AUC-ROC (appropriate for imbalanced binary classification)

### Model 2 — Random Forest Duration Regressor
- **Target**: `est_maintenance_hrs`
- **Stacking**: Uses risk probability (Model 1 output) as an additional feature
- **Metric**: R², MAE, RMSE with residual plots
- **Pitfall avoided**: `risk_label` never used as a feature (target leakage prevention)

---

## ⚙️ Optimizer — OR-Tools CP-SAT

**Constraints enforced:**
1. Each high-risk asset scheduled **at most once**
2. Only COA-**approved** blocks used
3. Maintenance duration ≤ block window duration
4. **Section matching**: asset must be in the same section as the block
5. **Power blocks**: only TDMS (OHE) assets — engineering blocks for TMS/SMMS
6. **Crew capacity**: max 3 assets per time block

**Objective**: Maximise total risk-score-weighted coverage

---

## 🚀 Quick Start

### 1. Setup environment
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Gemini API key
```bash
copy .env.example .env
# Open .env and add your Gemini API key
# Get a FREE key at: https://aistudio.google.com/app/apikey
```

### 3. Run the full pipeline
```bash
# Step 1: Generate synthetic data (~30 seconds)
python data_synthesizer.py

# Step 2: Train ML models (~2-4 minutes)
python ai_engine.py

# Step 3: Run optimizer (~1 minute)
python optimizer.py

# Step 4: Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open API documentation
Visit: **http://localhost:8000/docs** (Swagger UI)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | System health & model status |
| `GET`  | `/assets/risk` | All assets with risk scores (filter by system/zone) |
| `POST` | `/predict/risk` | Predict risk for a new asset |
| `GET`  | `/schedule/optimized` | Current OR-Tools maintenance schedule |
| `POST` | `/schedule/regenerate` | Re-run full pipeline in background |
| `GET`  | `/coa/blocks` | Approved COA maintenance windows |
| `GET`  | `/eda/summary` | Statistical summaries of all datasets |
| `POST` | `/copilot/query` | Gemini LLM natural language Q&A |

### Example: Predict risk for a TMS asset
```bash
curl -X POST http://localhost:8000/predict/risk \
  -H "Content-Type: application/json" \
  -d '{
    "asset_age_years": 18,
    "fault_count": 6,
    "last_inspection_days": 55,
    "component_wear_pct": 72,
    "traffic_load_mgt": 35,
    "zone": "NR",
    "source_system": "TMS"
  }'
```

### Example: Ask the Gemini copilot
```bash
curl -X POST http://localhost:8000/copilot/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which TDMS assets in NR zone need urgent attention this week?"}'
```

---

## 🗂️ Project Structure

```
Sih_Railway_project/
├── data/                        # Auto-generated CSV datasets
│   ├── tms_data.csv             # Track Management System records
│   ├── smms_data.csv            # Signalling Maintenance records
│   ├── tdms_data.csv            # Traction Distribution records
│   ├── coa_data.csv             # Control Office block availability
│   └── optimized_schedule.json  # OR-Tools output schedule
├── models/                      # Saved ML model artifacts
│   ├── risk_model.pkl           # XGBoost classifier + metadata
│   └── duration_model.pkl       # Random Forest regressor + metadata
├── reports/                     # EDA & evaluation plots
│   ├── eda_01_class_distribution.png
│   ├── eda_02_feature_distributions.png
│   ├── eda_03_correlation_heatmap.png
│   ├── eda_04_age_vs_fault.png
│   ├── eval_01_risk_classifier.png
│   └── eval_02_duration_regressor.png
├── data_synthesizer.py          # Step 1: Synthetic data generation
├── ai_engine.py                 # Step 2: ML model training
├── optimizer.py                 # Step 3: OR-Tools scheduling
├── api.py                       # Step 4: FastAPI + Gemini backend
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
└── README.md
```

---

## ⚠️ ML Pitfalls & How We Prevent Them

| Pitfall | Prevention |
|---------|------------|
| **Data Leakage** | Scaler/encoder fitted inside `Pipeline`, never on full dataset before split |
| **Class Imbalance** | XGBoost `scale_pos_weight = n_neg/n_pos`; Stratified K-Fold CV |
| **Target Leakage** | `risk_label` excluded from regression features; only `risk_probability` used |
| **Temporal Leakage** | COA blocks for future dates only; no past-block features in training |
| **Arbitrary Risk Labels** | Risk labels derived from domain physics, not random — ensures ML learns real patterns |

---

## 🔑 Getting a Free Gemini API Key

1. Go to **https://aistudio.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key and paste it in your `.env` file:
   ```
   GEMINI_API_KEY=AIza...your_key_here
   ```
5. The free tier supports **60 requests/minute** — more than enough for this project.
