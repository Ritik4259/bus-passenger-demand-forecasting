"""Tests for FastAPI inference service."""

from datetime import datetime
import pytest
from fastapi.testclient import TestClient

from src.app import app


@pytest.fixture(scope="module")
def client():
    """Create test client inside lifespan context so model is loaded into memory."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify healthcheck endpoint reports healthy status and loaded model."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "LightGBM"
    assert data["model_version"] == "1.0.0"


def test_predict_valid_schema(client: TestClient):
    """Verify POST /predict returns HTTP 200 and complies with required response schema."""
    payload = {
        "route_id": "LPU_JAL_101",
        "departure_datetime": "2026-07-01T08:30:00",
        "bus_capacity": 50,
        "is_raining": 0,
        "temperature_c": 20.0,
        "is_special_event": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["route_id"] == "LPU_JAL_101"
    assert data["bus_capacity"] == 50
    assert "predicted_passenger_demand" in data
    assert data["predicted_passenger_demand"] >= 0.0
    assert "predicted_occupancy_percentage" in data
    assert "action_flag" in data
    assert data["action_flag"] in ["UNDERUTILIZED", "OPTIMAL", "OVERFLOW_RISK"]
    assert "operational_recommendation" in data
    assert len(data["operational_recommendation"]) > 10
    assert "limitations" in data
    assert "Model assumes normal operating conditions" in data["limitations"]


def test_capacity_flag_shift_optimal_to_overflow(client: TestClient):
    """Test two different capacities on the same trip to prove action_flag shifts from OPTIMAL to OVERFLOW_RISK."""
    base_payload = {
        "route_id": "LPU_JAL_101",
        "departure_datetime": "2026-07-02T08:00:00",  # Morning peak on urban commuter
        "is_raining": 0,
        "temperature_c": 19.0,
    }

    # Case A: Adequate bus capacity (75 seats) -> should be OPTIMAL (or UNDERUTILIZED)
    payload_adequate = {**base_payload, "bus_capacity": 75}
    res_adequate = client.post("/predict", json=payload_adequate)
    assert res_adequate.status_code == 200
    data_adequate = res_adequate.json()

    # Case B: Heavily constrained bus capacity (15 seats) on morning rush hour -> must be OVERFLOW_RISK
    payload_constrained = {**base_payload, "bus_capacity": 15}
    res_constrained = client.post("/predict", json=payload_constrained)
    assert res_constrained.status_code == 200
    data_constrained = res_constrained.json()

    # Demand prediction should be identical for the same trip
    assert data_adequate["predicted_passenger_demand"] == data_constrained["predicted_passenger_demand"]

    # Occupancy percentage must scale inversely with capacity
    assert data_constrained["predicted_occupancy_percentage"] > data_adequate["predicted_occupancy_percentage"]
    assert data_constrained["predicted_occupancy_percentage"] > 90.0
    assert data_constrained["action_flag"] == "OVERFLOW_RISK"
    assert "High risk of passenger overflow" in data_constrained["operational_recommendation"]


def test_capacity_flag_underutilized(client: TestClient):
    """Verify that late night trip with large bus triggers UNDERUTILIZED flag."""
    payload = {
        "route_id": "LPU_JAL_101",
        "departure_datetime": "2026-07-02T23:30:00",  # Late night off-peak
        "bus_capacity": 80,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_occupancy_percentage"] < 50.0
    assert data["action_flag"] == "UNDERUTILIZED"
    assert "Low occupancy forecasted" in data["operational_recommendation"]


def test_cold_start_unknown_route(client: TestClient):
    """Verify that an unseen novel route falls back gracefully without raising an exception."""
    payload = {
        "route_id": "ROUTE_UNSEEN_999",
        "departure_datetime": "2026-07-03T10:00:00",
        "bus_capacity": 40,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_passenger_demand"] >= 0.0
    assert data["action_flag"] in ["UNDERUTILIZED", "OPTIMAL", "OVERFLOW_RISK"]


def test_predict_canonical_capacity_parameter(client: TestClient):
    """Verify that canonical 'capacity' parameter is accepted in place of 'bus_capacity'."""
    payload = {
        "route_id": "LPU_JAL_101",
        "departure_datetime": "2026-07-01T08:30:00",
        "capacity": 50,
        "is_raining": 0,
        "temperature_c": 20.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["capacity"] == 50
    assert data["bus_capacity"] == 50


def test_input_validation_errors(client: TestClient):
    """Verify Pydantic rejects invalid inputs with 422 Unprocessable Entity."""
    # Invalid capacity <= 0
    res1 = client.post(
        "/predict",
        json={"route_id": "LPU_JAL_101", "departure_datetime": "2026-07-01T08:00:00", "bus_capacity": 0},
    )
    assert res1.status_code == 422

    # Malformed datetime
    res2 = client.post(
        "/predict",
        json={"route_id": "LPU_JAL_101", "departure_datetime": "not-a-datetime", "bus_capacity": 50},
    )
    assert res2.status_code == 422

    # Missing required field (route_id)
    res3 = client.post(
        "/predict",
        json={"departure_datetime": "2026-07-01T08:00:00", "bus_capacity": 50},
    )
    assert res3.status_code == 422
