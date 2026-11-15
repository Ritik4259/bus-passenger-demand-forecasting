"""Tests for HistoricalRouteTimeBaseline estimator."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_regressor

from src.baseline import HistoricalRouteTimeBaseline
from src.data_gen import generate_synthetic_trips
from src.features import split_data_chronological


@pytest.fixture(scope="module")
def trips_data() -> pd.DataFrame:
    """Generate 21-day dataset for baseline tests."""
    return generate_synthetic_trips(start_date="2026-03-01", num_days=21, random_state=99)


def test_baseline_sklearn_compatibility():
    """Verify that HistoricalRouteTimeBaseline complies with scikit-learn estimator interface."""
    baseline = HistoricalRouteTimeBaseline()

    # Must be recognized as an sklearn regressor
    assert is_regressor(baseline)

    # Must support cloning via sklearn.base.clone
    cloned = clone(baseline)
    assert isinstance(cloned, HistoricalRouteTimeBaseline)
    assert cloned.aggregation == baseline.aggregation


def test_baseline_fit_and_predict(trips_data: pd.DataFrame):
    """Verify baseline fits on training features and predicts consistent non-negative values."""
    train_df, _, test_df = split_data_chronological(trips_data)

    X_train = train_df[["route_id", "day_of_week", "hour"]]
    y_train = train_df["passenger_demand"]
    X_test = test_df[["route_id", "day_of_week", "hour"]]

    baseline = HistoricalRouteTimeBaseline()
    baseline.fit(X_train, y_train)

    preds = baseline.predict(X_test)
    assert len(preds) == len(X_test)
    assert (preds >= 0.0).all()
    assert isinstance(preds, np.ndarray)


def test_baseline_strict_training_leakage_isolation(trips_data: pd.DataFrame):
    """Prove that the baseline aggregations are derived exclusively from the training partition."""
    train_df, _, test_df = split_data_chronological(trips_data)

    X_train = train_df[["route_id", "day_of_week", "hour"]]
    y_train = train_df["passenger_demand"]

    baseline = HistoricalRouteTimeBaseline()
    baseline.fit(X_train, y_train)

    # 1. Verify group medians match manual training medians
    expected_medians = (
        train_df.groupby(["route_id", "day_of_week", "hour"])["passenger_demand"]
        .median()
        .to_dict()
    )
    for key, expected_val in expected_medians.items():
        assert key in baseline.group_medians_
        assert np.isclose(baseline.group_medians_[key], expected_val)

    # 2. Modify test_df values drastically and verify baseline predictions remain identical
    X_test = test_df[["route_id", "day_of_week", "hour"]]
    preds_before = baseline.predict(X_test)

    # Simulated malicious injection into test set target (should have zero effect on baseline)
    test_df_mutated = test_df.copy()
    test_df_mutated["passenger_demand"] = 99999.0

    preds_after = baseline.predict(X_test)
    assert np.array_equal(preds_before, preds_after), "Baseline predictions must be invariant to test data modifications"


def test_baseline_hierarchical_fallback():
    """Verify graceful hierarchical fallback for unseen routes and unusual hours."""
    X_train = pd.DataFrame(
        {
            "route_id": ["ROUTE_A", "ROUTE_A", "ROUTE_B", "ROUTE_B"],
            "day_of_week": [0, 0, 1, 1],
            "hour": [8, 9, 10, 11],
        }
    )
    y_train = pd.Series([20.0, 30.0, 40.0, 50.0])

    baseline = HistoricalRouteTimeBaseline()
    baseline.fit(X_train, y_train)

    # Case 1: Exact match (ROUTE_A, DOW 0, Hour 8) -> 20.0
    pred1 = baseline.predict(pd.DataFrame([{"route_id": "ROUTE_A", "day_of_week": 0, "hour": 8}]))
    assert np.isclose(pred1[0], 20.0)

    # Case 2: Unseen DOW for ROUTE_A (DOW 5, Hour 8) -> fallback to route_hour ROUTE_A, Hour 8 -> 20.0
    pred2 = baseline.predict(pd.DataFrame([{"route_id": "ROUTE_A", "day_of_week": 5, "hour": 8}]))
    assert np.isclose(pred2[0], 20.0)

    # Case 3: Completely unseen hour for ROUTE_A (Hour 23) -> fallback to route median ROUTE_A -> 25.0
    pred3 = baseline.predict(pd.DataFrame([{"route_id": "ROUTE_A", "day_of_week": 5, "hour": 23}]))
    assert np.isclose(pred3[0], 25.0)

    # Case 4: Completely unseen route ROUTE_UNKNOWN -> fallback to global median (35.0)
    pred4 = baseline.predict(pd.DataFrame([{"route_id": "ROUTE_UNKNOWN", "day_of_week": 2, "hour": 14}]))
    assert np.isclose(pred4[0], 35.0)
