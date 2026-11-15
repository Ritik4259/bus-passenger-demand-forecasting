# 🚌 Bus Passenger Demand Forecasting & Fleet Decision Support

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-19%20passed-brightgreen.svg)](tests/)

A production-ready, reproducible Machine Learning decision-support solution to forecast bus passenger demand by route and departure timestamp **24 hours prior to departure ($T - 24\text{h}$)**. Designed for transit dispatchers and scheduling operators to make human-in-the-loop decisions on vehicle sizing and overflow prevention.

---

## 🌟 Key Highlights

- **Target Definition**: Models **unconstrained latent passenger demand** rather than capacity-censored boardings, enabling the detection of true overflow risks. ([ADR-0001](docs/adr/0001-unconstrained-demand-target.md)).
- **Zero Temporal Leakage**: Enforces strict chronological data splitting (70% Train, 15% Val, 15% Test) and a strict $T - 24\text{h}$ feature cutoff rule for all historical lag statistics ([ADR-0002](docs/adr/0002-leakage-safe-24h-feature-store.md)).
- **Exogenous Demand Architecture**: Vehicle capacity is strictly excluded from regressor input features to prevent reverse causality ([ADR-0004](docs/adr/0004-exogenous-demand-capacity-exclusion.md)).
- **Benchmarked Models**: Evaluates a scikit-learn compatible `HistoricalRouteTimeBaseline`, regularized `RidgeRegressionPipeline`, and gradient boosted `LightGBMPipeline`.
- **Production Artifact**: Encapsulates preprocessing transformations, cyclical datetime encoders, and estimator inside a unified `.joblib` pipeline.
- **FastAPI Endpoint**: Validated REST API (`POST /predict`, `GET /health`) returning demand, occupancy rate, capacity action flags, and limitations.
- **Streamlit Demo UI**: Interactive dispatcher interface featuring live what-if fleet sizing simulations.
- **100% Test Coverage**: 19 automated Pytest integration and regression tests.

---

## 📁 Repository Structure

```
Assignment/
├── SPEC.md                               # System specification and architecture constraints
├── GLOSSARY.md                           # Canonical transit domain glossary
├── AGENTS.md                             # Agent operating guidelines
├── requirements.txt                      # Python dependencies
├── .env.example                          # Environment variable template
├── docs/
│   ├── adr/                              # Architectural Decision Records
│   │   ├── 0001-unconstrained-demand-target.md
│   │   ├── 0002-leakage-safe-24h-feature-store.md
│   │   ├── 0003-unified-pipeline-serialization.md
│   │   └── 0004-exogenous-demand-capacity-exclusion.md
│   └── agents/                           # Engineering skill guidelines
├── src/
│   ├── __init__.py
│   ├── data_gen.py                       # Synthetic trip dataset generator (42,300 trips)
│   ├── features.py                       # Chronological split, cyclical & 24h lag pipeline
│   ├── baseline.py                       # Scikit-learn HistoricalRouteTimeBaseline estimator
│   ├── train.py                          # Pipeline benchmarking and model serialization
│   ├── evaluate.py                       # Granular evaluation (peak/off-peak, per-route)
│   └── app.py                            # Production FastAPI inference service
├── app/
│   └── ui.py                             # Interactive Streamlit dispatcher demo UI
├── tests/
│   ├── test_data_gen.py                  # Censoring, schema, and diurnal curve tests
│   ├── test_features.py                  # Cryptographic non-overlapping split & 24h lag tests
│   ├── test_baseline.py                  # Sklearn compatibility, training isolation & fallbacks
│   └── test_api.py                       # FastAPI TestClient schema and flag transition tests
├── models/
│   └── bus_demand_model.joblib           # Production model artifact + bundled lookup table
├── artifacts/
│   └── evaluation_report.json            # Structured evaluation and diagnostics report
└── report/
    └── technical_report.md               # Comprehensive rubric technical report
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd Assignment

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
```

### 2. Run Automated Test Suite (19/19 Passing)

Verify end-to-end integrity, temporal leakage prevention, and API contracts:
```bash
pytest tests/ -v
```

### 3. Reproduce Data, Training & Evaluation from Scratch

```bash
# Step A: Generate 42,300 synthetic trips (180 days)
python -m src.data_gen --days 180 --output data/synthetic_trips.csv --seed 42

# Step B: Train baseline, Ridge, and LightGBM pipelines
python -m src.train --data data/synthetic_trips.csv --models-dir models

# Step C: Run granular evaluation and generate artifacts/evaluation_report.json
python -m src.evaluate --data data/synthetic_trips.csv --model models/bus_demand_model.joblib
```

---

## 🌐 Running the Inference API & Demo UI

### Launch FastAPI Server

```bash
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

#### Sample Query:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "route_id": "LPU_JAL_101",
       "departure_datetime": "2026-08-10T08:00:00",
       "bus_capacity": 50,
       "is_raining": 1,
       "temperature_c": 28.5
     }'
```

#### Sample Response:
```json
{
  "route_id": "LPU_JAL_101",
  "departure_datetime": "2026-08-10T08:00:00",
  "bus_capacity": 50,
  "predicted_passenger_demand": 64.2,
  "predicted_occupancy_percentage": 128.4,
  "action_flag": "OVERFLOW_RISK",
  "operational_recommendation": "High risk of passenger overflow and denied boardings. Dispatch supplemental relief bus or upsize vehicle to high-capacity articulated fleet.",
  "model_name": "LightGBM",
  "model_version": "1.0.0",
  "limitations": "Model assumes normal operating conditions; does not account for real-time weather shocks or novel events. Predictions represent unconstrained passenger demand 24 hours prior to departure."
}
```

### Launch Interactive Streamlit Dispatcher UI

In a separate terminal:
```bash
streamlit run app/ui.py
```
Opens interactive dashboard at [http://localhost:8501](http://localhost:8501).

---

## 📊 Benchmark Summary

Evaluated on 6,345 chronologically isolated test trips:

| Candidate Model | Val MAE | Val RMSE | Val WAPE | Test MAE | Test RMSE | Test WAPE | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Baseline** | 7.681 | 11.733 | 28.61% | 6.558 | 9.373 | 27.52% | Benchmark |
| **Ridge Regression Pipeline** | 9.680 | 14.053 | 36.05% | 8.827 | 12.022 | 37.04% | Underfitting non-linear peaks |
| **LightGBM Pipeline** | **7.204** | **10.554** | **26.83%** | **6.309** | **9.097** | **26.48%** | **Production Selected** |

### Granular Test Evaluation
- **Peak Hours (7-10 AM & 5-8 PM)**: MAE 7.62 | RMSE 10.87 | **WAPE 25.46%**
- **Off-Peak Hours**: MAE 5.35 | RMSE 7.54 | **WAPE 27.62%**
- **Action Flag Accuracy**: **78.88%**
- **Overflow Risk Recall**: **55.29%** (detects major capacity surges 24h in advance)

---

## ⚠️ Known Incomplete Features & Assumptions

As required for submission transparency:

1. **24-Hour Exogenous Weather Assumptions**: The model assumes forecasted temperature and precipitation values provided at $T - 24\text{h}$ are accurate. Unpredicted sudden flash storms occurring minutes before departure are not captured.
2. **Cold-Start for Novel Routes**: For newly introduced transit lines without historical telemetry, the service gracefully falls back to the global training median (25.0 passengers). Operational guidelines require dispatchers to flag new routes as probationary for the first 14 days.
3. **Single-Vehicle Trip Independence**: Each scheduled trip is evaluated independently. Dynamic vehicle chaining (e.g. Bus A finishing Route 101 and turning around to serve Route 202) is handled by the human dispatcher, not automatically solved by the ML regressor.
4. **No Passenger Substitution / Cannibalization**: The model treats demand on parallel routes as independent. If Route 101 is canceled, the model does not automatically re-route passengers onto Route 202.
5. **Human-in-the-Loop Requirement**: The service functions as a decision advisor. Automated service cancellation or vehicle reassignment without dispatcher verification is explicitly prohibited.

---

## 📄 License & Compliance

MIT License. Synthetic dataset is free of Personally Identifiable Information (PII) and compliant with GDPR/CCPA.
