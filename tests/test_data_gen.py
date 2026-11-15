"""Tests for synthetic data generation module."""

import numpy as np
import pandas as pd
import pytest

from src.data_gen import generate_synthetic_trips


@pytest.fixture(scope="module")
def sample_dataset() -> pd.DataFrame:
    """Generate a small 14-day dataset for testing."""
    return generate_synthetic_trips(
        start_date="2026-03-01",
        num_days=14,
        random_state=123,
    )


def test_schema_and_columns(sample_dataset: pd.DataFrame):
    """Verify that all required columns are present with expected non-empty values."""
    required_cols = [
        "trip_id",
        "route_id",
        "route_name",
        "departure_datetime",
        "capacity",
        "is_weekend",
        "day_of_week",
        "hour",
        "minute",
        "month",
        "is_holiday",
        "is_raining",
        "temperature_c",
        "is_special_event",
        "passenger_demand",
        "boarded_passengers",
        "occupancy_rate",
        "action_flag",
    ]
    for col in required_cols:
        assert col in sample_dataset.columns, f"Missing required column: {col}"
    assert len(sample_dataset) > 0


def test_train_arrival_surges_phg_lpu(sample_dataset: pd.DataFrame):
    """Verify Sunday evening and Monday morning train spikes on PHG_LPU_202."""
    railway_trips = sample_dataset[sample_dataset["route_id"] == "PHG_LPU_202"]

    # Sunday evening peak train arrivals (DOW 6, hour between 18 and 21)
    sun_eve = railway_trips[(railway_trips["day_of_week"] == 6) & (railway_trips["hour"].isin([18, 19, 20, 21]))]
    # Sunday midday (DOW 6, hour between 11 and 14)
    sun_mid = railway_trips[(railway_trips["day_of_week"] == 6) & (railway_trips["hour"].isin([11, 12, 13, 14]))]

    assert sun_eve["passenger_demand"].mean() > sun_mid["passenger_demand"].mean(), (
        "Sunday evening train arrivals must produce higher passenger demand than Sunday midday"
    )


def test_chronological_ordering(sample_dataset: pd.DataFrame):
    """Verify dataset is strictly sorted in chronological order."""
    dates = pd.to_datetime(sample_dataset["departure_datetime"])
    assert dates.is_monotonic_increasing, "Trips must be strictly sorted by departure_datetime"


def test_capacity_and_censoring_logic(sample_dataset: pd.DataFrame):
    """Verify boarded passengers never exceed bus capacity, while latent demand can."""
    assert (sample_dataset["boarded_passengers"] <= sample_dataset["capacity"]).all(), (
        "Boarded passengers must be censored at vehicle capacity"
    )
    assert (sample_dataset["passenger_demand"] >= sample_dataset["boarded_passengers"]).all(), (
        "Latent passenger demand must be >= boarded passengers"
    )
    assert (sample_dataset["passenger_demand"] >= 0).all(), "Passenger demand must be non-negative"


def test_action_flag_thresholds(sample_dataset: pd.DataFrame):
    """Verify that action flags strictly correspond to the occupancy rate thresholds."""
    under = sample_dataset[sample_dataset["occupancy_rate"] < 0.50]
    optimal = sample_dataset[
        (sample_dataset["occupancy_rate"] >= 0.50) & (sample_dataset["occupancy_rate"] <= 0.90)
    ]
    overflow = sample_dataset[sample_dataset["occupancy_rate"] > 0.90]

    assert (under["action_flag"] == "UNDERUTILIZED").all()
    assert (optimal["action_flag"] == "OPTIMAL").all()
    assert (overflow["action_flag"] == "OVERFLOW_RISK").all()


def test_diurnal_peak_effects(sample_dataset: pd.DataFrame):
    """Verify that rush hour demand on urban commuter route is significantly higher than off-peak."""
    commuter_trips = sample_dataset[sample_dataset["route_id"] == "LPU_JAL_101"]
    weekday_commuter = commuter_trips[commuter_trips["is_weekend"] == 0]

    peak_trips = weekday_commuter[weekday_commuter["hour"].isin([8, 17])]
    offpeak_trips = weekday_commuter[weekday_commuter["hour"].isin([5, 22])]

    assert peak_trips["passenger_demand"].mean() > offpeak_trips["passenger_demand"].mean() * 2.0, (
        "Peak hour demand should be substantially higher than early morning / late night"
    )
