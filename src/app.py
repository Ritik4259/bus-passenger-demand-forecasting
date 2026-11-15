"""FastAPI production inference service for Bus Passenger Demand Forecasting.

Loads unified Pipeline and historical lag lookup from joblib.
Provides:
- POST /predict: Predicts latent passenger demand, seat occupancy, and action flag.
- GET /health: Healthcheck and model readiness probe.
"""

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from src.domain import ActionFlag, compute_action_flag
from src.features import FEATURE_COLUMNS, is_punjab_holiday

# Global state holder for in-memory model and historical lookup dictionary
model_state: Dict[str, Any] = {}

MODEL_PATH = os.getenv("MODEL_PATH", "models/bus_demand_model.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact and historical lookup table into memory during startup."""
    artifact_file = Path(MODEL_PATH)
    if not artifact_file.exists():
        raise RuntimeError(
            f"Model artifact not found at {artifact_file}. Train model first via `python -m src.train`."
        )

    print(f"Loading production model artifact from {artifact_file}...")
    bundle = joblib.load(artifact_file)
    model_state["pipeline"] = bundle["pipeline"]
    model_state["lag_lookup_dict"] = bundle["lag_lookup_dict"]
    model_state["model_name"] = bundle.get("model_name", "LightGBM")
    model_state["model_version"] = bundle.get("model_version", "1.0.0")
    model_state["limitations"] = bundle.get(
        "limitations",
        [
            "Model assumes normal operating conditions; does not account for real-time weather shocks or novel events.",
            "Predictions represent unconstrained passenger travel intent 24 hours prior to departure.",
        ],
    )
    print(f"Model {model_state['model_name']} v{model_state['model_version']} loaded successfully!")
    yield
    model_state.clear()
    print("Model service shut down and memory cleared.")


app = FastAPI(
    title="Bus Passenger Demand Forecasting API",
    description="24-hour lookahead passenger demand forecasting and capacity decision support service.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """Input payload for trip demand prediction."""

    route_id: str = Field(
        ...,
        description="Unique route identifier (e.g. LPU_JAL_101, PHG_LPU_202).",
        examples=["LPU_JAL_101"],
    )
    departure_datetime: datetime = Field(
        ...,
        description="Scheduled departure timestamp in ISO 8601 format.",
        examples=["2026-07-01T08:30:00"],
    )
    bus_capacity: Optional[int] = Field(
        default=None,
        ge=1,
        le=300,
        description="Vehicle seat capacity assigned to this departure.",
        examples=[50],
    )
    capacity: Optional[int] = Field(
        default=None,
        ge=1,
        le=300,
        description="Canonical domain capacity parameter.",
        examples=[50],
    )
    is_raining: Optional[int] = Field(
        default=0,
        ge=0,
        le=1,
        description="Weather precipitation indicator (1 = rain, 0 = dry).",
    )
    temperature_c: Optional[float] = Field(
        default=18.0,
        description="Forecasted temperature in Celsius at departure time.",
    )
    is_special_event: Optional[int] = Field(
        default=0,
        ge=0,
        le=1,
        description="Special city event indicator affecting demand.",
    )

    @model_validator(mode="after")
    def validate_capacity_present(self):
        cap = self.capacity if self.capacity is not None else self.bus_capacity
        if cap is None:
            raise ValueError("Either 'capacity' or 'bus_capacity' must be provided.")
        return self

    @property
    def effective_capacity(self) -> int:
        return self.capacity if self.capacity is not None else (self.bus_capacity or 50)


class PredictResponse(BaseModel):
    """Output payload containing demand forecast and operational decision flags."""

    route_id: str
    departure_datetime: datetime
    bus_capacity: int
    capacity: int
    predicted_passenger_demand: float
    predicted_occupancy_percentage: float
    action_flag: str
    operational_recommendation: str
    model_name: str
    model_version: str
    limitations: str


class HealthResponse(BaseModel):
    """Service status response."""

    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
def healthcheck():
    """Verify service readiness and model loading status."""
    is_loaded = "pipeline" in model_state
    return HealthResponse(
        status="healthy" if is_loaded else "uninitialized",
        model_loaded=is_loaded,
        model_name=model_state.get("model_name"),
        model_version=model_state.get("model_version"),
    )


def extract_inference_features(
    req: PredictRequest,
    lag_lookup: Dict[str, Any],
) -> pd.DataFrame:
    """Transform raw request parameters and extract cyclical time and lookup lag features."""
    dt = req.departure_datetime
    hour_float = dt.hour + dt.minute / 60.0
    day_of_week = dt.weekday()
    is_weekend = int(day_of_week in (5, 6))
    month = dt.month
    is_holiday = is_punjab_holiday(dt)

    # Cyclical representations
    hour_sin = np.sin(2.0 * np.pi * hour_float / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour_float / 24.0)
    dow_sin = np.sin(2.0 * np.pi * day_of_week / 7.0)
    dow_cos = np.cos(2.0 * np.pi * day_of_week / 7.0)
    month_sin = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    month_cos = np.cos(2.0 * np.pi * (month - 1) / 12.0)

    # Hierarchical lag retrieval strictly from historical training lookup
    route_dow_hour_key = f"{req.route_id}_DOW{day_of_week}_H{dt.hour}"
    route_hour_key = f"{req.route_id}_H{dt.hour}"

    dow_hour_stats = lag_lookup.get("route_dow_hour_stats", {})
    hour_stats = lag_lookup.get("route_hour_stats", {})
    route_stats = lag_lookup.get("route_stats", {})
    global_stats = lag_lookup.get("global_stats", {})
    global_med = lag_lookup.get("global_median", 25.0)

    if route_dow_hour_key in dow_hour_stats:
        matched = dow_hour_stats[route_dow_hour_key]
    elif route_hour_key in hour_stats:
        matched = hour_stats[route_hour_key]
    elif req.route_id in route_stats:
        matched = route_stats[req.route_id]
    else:
        matched = global_stats

    # Extract distinct calibrated historical lags (eliminates train-serve lag skew)
    lag_24h = matched.get("lag_24h_median", global_stats.get("lag_24h_median", global_med))
    lag_7d = matched.get("lag_7d_median", global_stats.get("lag_7d_median", global_med))
    rolling_7d = matched.get("rolling_7d_median", global_stats.get("rolling_7d_median", global_med))
    rolling_14d = matched.get("rolling_14d_median", global_stats.get("rolling_14d_median", global_med))

    row = {
        "route_id": req.route_id,
        "is_weekend": is_weekend,
        "day_of_week": day_of_week,
        "hour": dt.hour,
        "minute": dt.minute,
        "month": month,
        "is_holiday": is_holiday,
        "is_raining": req.is_raining or 0,
        "temperature_c": req.temperature_c if req.temperature_c is not None else 18.0,
        "is_special_event": req.is_special_event or 0,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "lag_24h_demand": lag_24h,
        "lag_7d_demand": lag_7d,
        "rolling_7d_mean": rolling_7d,
        "rolling_14d_mean": rolling_14d,
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.post("/predict", response_model=PredictResponse)
def predict_passenger_demand(request: PredictRequest):
    """Predict unconstrained passenger demand, occupancy rate, and operational capacity flag."""
    if "pipeline" not in model_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check /health status.",
        )

    pipeline = model_state["pipeline"]
    lag_lookup = model_state["lag_lookup_dict"]
    effective_capacity = request.effective_capacity

    # 1. Feature extraction with zero leakage
    X_input = extract_inference_features(request, lag_lookup)

    # 2. Predict unconstrained demand
    raw_pred = pipeline.predict(X_input)
    predicted_demand = float(np.maximum(0.0, raw_pred[0]))
    predicted_demand = round(predicted_demand, 1)

    # 3. Calculate predicted occupancy percentage
    occupancy_rate = predicted_demand / float(effective_capacity)
    occupancy_pct = round(occupancy_rate * 100.0, 1)

    # 4. Canonical action flag and Transport Coordinator recommendation via shared domain logic
    action_flag_enum, recommendation = compute_action_flag(predicted_demand, effective_capacity)

    limitations_text = (
        "Model assumes normal operating conditions; does not account for real-time weather shocks "
        "or novel events. Predictions represent unconstrained passenger demand 24 hours prior to departure."
    )

    return PredictResponse(
        route_id=request.route_id,
        departure_datetime=request.departure_datetime,
        bus_capacity=effective_capacity,
        capacity=effective_capacity,
        predicted_passenger_demand=predicted_demand,
        predicted_occupancy_percentage=occupancy_pct,
        action_flag=action_flag_enum.value,
        operational_recommendation=recommendation,
        model_name=model_state["model_name"],
        model_version=model_state["model_version"],
        limitations=limitations_text,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
