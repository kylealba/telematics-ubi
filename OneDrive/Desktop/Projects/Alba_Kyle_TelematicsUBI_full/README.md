# Telematics Integration in Auto Insurance – POC (Full Code)

This repository is a proof-of-concept (POC) for a telematics-based, usage-based insurance (UBI) system.
It demonstrates: data ingestion (simulated), near-real-time processing, behavior-based risk scoring, a pricing engine API, and a simple user dashboard.

## Architecture (POC)
- **Ingest:** Simulated GPS/accelerometer and trip metadata → parquet/csv.
- **Process:** Batch/near-real-time feature aggregation (per trip/per driver).
- **Model:** Gradient-boosted trees for behavior-based risk scoring (0–1) + isotonic calibration.
- **Pricing Engine API:** FastAPI service that converts risk score → price factor.
- **Dashboard:** Streamlit app for user transparency & engagement.
- **Security/Privacy (POC):** Token-auth placeholder; field-level PII minimization; anonymized sample data.

See `docs/architecture.md` for the component view and data schemas.

---

## 1) Setup

### Prereqs
- Python 3.10+
- pip / venv (recommended)

### Create & activate a virtualenv
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate
pip install -U pip
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 2) Run the POC

### (A) Generate synthetic telematics data
```bash
python src/ingest/simulator.py --n_drivers 200 --days 30 --out data/sample_events.parquet
```

### (B) Build features (trip/driver aggregates)
```bash
python src/processing/feature_pipeline.py --events data/sample_events.parquet --out data/driver_features.csv
```

### (C) Train the risk model
```bash
python src/models/train.py --features data/driver_features.csv --model models/risk_model.pkl
```

### (D) Start the pricing API
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```
- Example request:
```bash
curl -X POST http://localhost:8000/score   -H "Content-Type: application/json"   -d '{"driver_id":"D-001","mean_speed":47.2,"harsh_brake_rate":0.08,"night_miles_pct":0.35,"urban_miles_pct":0.6,"accel_var":1.2,"phone_use_pct":0.05,"total_miles":800}'
```

### (E) Launch the dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 3) Evaluate
```bash
python src/models/evaluate.py --features data/driver_features.csv --model models/risk_model.pkl --out docs/eval_report.json
```
Outputs ROC-AUC, PR-AUC, calibration, and feature importance. See `docs/eval_report.json` and dashboard "Model" tab.

---

## 4) Notes
- **Models:** scikit-learn GradientBoostingClassifier (POC); swap with XGBoost/LightGBM for scale.
- **External services:** None required. Replace simulator with hardware/mobile SDK collectors for production.
- **Privacy:** Sample data is fully synthetic; no real PII. In production, enforce consent, data minimization, encryption at rest & in transit, and jurisdiction-specific policies (CCPA/GDPR).

---

## Repo layout
```
/src
  /ingest              # data simulators & adapters
  /processing          # feature builders
  /models              # training, evaluation
  /api                 # FastAPI pricing engine
  /dashboard           # Streamlit user UI
/models                # saved weights
/docs                  # design docs & diagrams
/bin                   # helper run scripts
/data                  # sample & intermediate data
```
