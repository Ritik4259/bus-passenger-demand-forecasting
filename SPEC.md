# System Specification: LPU & Punjab Regional Bus Demand Forecasting & Fleet Decision Support

## 1. Executive Summary & Problem Statement
This system provides an end-to-end, reproducible machine learning solution to forecast bus passenger demand along the **Phagwara – Jalandhar – Lovely Professional University (LPU) NH-44 corridor** in Punjab, India. Predictions are generated 24 hours prior to departure ($T - 24\text{h}$) to assist university transport coordinators and regional dispatchers in human-in-the-loop decisions regarding bus sizing, service consolidation, or dispatching emergency relief coaches to avoid stranding students and daily commuters.

## 2. Route Topography & Operational Profiles

| Route ID | Corridor Name | Default Capacity | Stops / Landmarks | Primary Demand Dynamics |
| :--- | :--- | :---: | :--- | :--- |
| **`LPU_JAL_101`** | LPU Main Gate → Jalandhar City Bus Stand | 50 (Standard) | LPU Gate 1, Rama Mandi, PAP Chowk, BMC Chowk, Jalandhar Bus Stand | Heavy college commute peaks (08:00 AM & 05:30 PM). |
| **`PHG_LPU_202`** | Phagwara Jn Railway Station → LPU Campus | 40 (Mini-coach) | Phagwara Railway Station, Goal Chowk, Law Gate, LPU Uni-Mall | Massive surges on Sunday evenings and Monday mornings from train arrivals. |
| **`LPU_MDL_303`** | LPU Campus → Model Town, Jalandhar | 45 (AC Coach) | LPU Main Gate, Cantt Road, Geeta Mandir, Model Town Market | Weekend & Friday evening leisure and dining peaks (04:00 PM – 09:00 PM). |
| **`INT_CAMPUS_404`**| LPU Main Gate → Block 34 / Uni-Hospital | 60 (Transit) | Main Gate, Uni-Hospital, Block 14, Central Library, Block 34 | Intense academic class-change bursts at 08:45 AM, 01:00 PM, and 05:00 PM. |
| **`JAL_ASR_505`** | Jalandhar Cantt → Phagwara GT Road Feeder | 35 (Minibus) | Jalandhar Cantt Station, Chaheru, Chiheru Bridge, Phagwara GT Road | Steady, sparse baseline connecting suburban commuters. |

## 3. System Architecture & Decisions

### 3.1 Target Variable & Latent Demand
- **Target Variable**: `passenger_demand` (float/int), representing true unconstrained latent passenger demand seeking to board.
- **Constraints Recorded**: `capacity` (int) and `boarded_passengers` = $\min(\text{passenger\_demand}, \text{capacity})$.
- **Design Rationale**: Censoring demand at vehicle capacity prevents learning true peak surges, blinding the system to overflow risks (see `docs/adr/0001-unconstrained-demand-target.md`).

### 3.2 Leakage-Safe Chronological Pipeline
- **Split Strategy**: Strict chronological split (70% Train, 15% Val, 15% Test).
- **24-Hour Horizon Cutoff**: All temporal lag and rolling statistical features (e.g., 24h lag, 7-day route lag, 14-day rolling mean) for a trip departing at time $T$ must be computed exclusively from observations where departure $\le T - 24\text{h}$ (see `docs/adr/0002-leakage-safe-24h-feature-store.md`).
- **Exogenous Demand Rule**: Vehicle capacity is strictly excluded from regressor input features to prevent reverse causality (see `docs/adr/0004-exogenous-demand-capacity-exclusion.md`).

### 3.3 Operational Action Flags & Decision Logic
The operational recommendation is determined by predicted occupancy:
$$\text{Occupancy Rate} = \frac{\widehat{\text{demand}}}{\text{capacity}}$$

| Occupancy Rate | Action Flag | Operational Guidance |
| :--- | :--- | :--- |
| $< 50\%$ | `UNDERUTILIZED` | Downsize to minibus or consolidate scheduled runs to save fuel. |
| $50\% - 90\%$ | `OPTIMAL` | Standard fleet assignment adequate; balanced passenger load. |
| $> 90\%$ | `OVERFLOW_RISK` | High probability of stranded students; dispatch supplemental relief bus or upsize vehicle. |

### 3.4 Model Benchmarking & Serialization
- **Models**:
  1. `HistoricalRouteTimeBaseline`: Custom scikit-learn compatible estimator calculating median training demand per `(route_id, day_of_week, departure_hour)`.
  2. `RidgeRegressionPipeline`: Linear model with one-hot encoded route, scaled continuous features, and cyclical datetime encodings.
  3. `LightGBMPipeline`: Gradient boosted trees with route categorical handling, calendar/holiday features, and 24h-lag features.
- **Metrics**: MAE, RMSE, WAPE ($\frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$).
- **Artifact**: Single `.joblib` file storing the complete scikit-learn `Pipeline` and training lookup dictionary.

### 3.5 Production Interfaces & UI/UX
- **REST API (FastAPI)**:
  - `POST /predict`: Pydantic-validated endpoint returning unconstrained demand, occupancy percentage, action flag, and limitations.
  - `GET /health`: Model status and readiness probe.
- **Interactive UI (Streamlit)**:
  - Interactive **Plotly 24-Hour Demand Curve** with capacity threshold overlay.
  - Interactive **Pydeck Geospatial Route Map** of Phagwara/Jalandhar corridor coordinates.
  - Visual **Seat Occupancy Breakdown** (Seated vs. Overflow vs. Empty) and live FastAPI JSON payload inspector.
