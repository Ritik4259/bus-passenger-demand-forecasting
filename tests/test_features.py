"""Tests for feature engineering, chronological splitting, and leakage safe lags."""

import numpy as np
import pandas as pd
import pytest

from src.data_gen import generate_synthetic_trips
from src.features import (
    add_cyclical_time_features,
    add_strict_24h_lag_features,
    build_historical_lag_lookup,
    split_data_chronological,
)


@pytest.fixture(scope="module")
def trips_df() -> pd.DataFrame:
    """Generate 28-day sample dataset for feature tests."""
    return generate_synthetic_trips(
        start_date="2026-02-01",
        num_days=28,
        random_state=42,
    )


def test_chronological_split_strict_non_overlapping(trips_df: pd.DataFrame):
    """Prove cryptographically and temporally that train, val, and test splits never overlap in time."""
    train_df, val_df, test_df = split_data_chronological(
        trips_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # 1. Total row count conservation
    assert len(train_df) + len(val_df) + len(test_df) == len(trips_df)

    # 2. Strict chronological boundaries
    max_train_time = train_df["departure_datetime"].max()
    min_val_time = val_df["departure_datetime"].min()
    max_val_time = val_df["departure_datetime"].max()
    min_test_time = test_df["departure_datetime"].min()

    assert max_train_time < min_val_time, (
        f"Train timeline ({max_train_time}) must be strictly prior to Val timeline ({min_val_time})"
    )
    assert max_val_time < min_test_time, (
        f"Val timeline ({max_val_time}) must be strictly prior to Test timeline ({min_test_time})"
    )

    # 3. Disjoint set of timestamps
    train_dates = set(train_df["departure_datetime"])
    val_dates = set(val_df["departure_datetime"])
    test_dates = set(test_df["departure_datetime"])

    assert train_dates.isdisjoint(val_dates), "Train and Val contain overlapping departure timestamps"
    assert val_dates.isdisjoint(test_dates), "Val and Test contain overlapping departure timestamps"
    assert train_dates.isdisjoint(test_dates), "Train and Test contain overlapping departure timestamps"


def test_strict_24h_lag_leakage_safe(trips_df: pd.DataFrame):
    """Prove that all historical lag features strictly observe a >= 24h separation from departure time."""
    df_lags = add_strict_24h_lag_features(trips_df)

    # Check lag_time_24h vs departure_datetime
    valid_lags = df_lags.dropna(subset=["lag_24h_demand", "lag_time_24h"]).copy()
    assert len(valid_lags) > 0

    time_diff = (valid_lags["departure_datetime"] - valid_lags["lag_time_24h"]).dt.total_seconds() / 3600.0

    # Every single matched trip must be at least 24.0 hours prior to the current departure
    assert (time_diff >= 24.0).all(), (
        f"Temporal leakage detected! Min time difference is {time_diff.min()} hours (expected >= 24.0h)"
    )


def test_cyclical_time_features(trips_df: pd.DataFrame):
    """Verify trigonometric properties of cyclical features (sin^2 + cos^2 == 1)."""
    df_cyc = add_cyclical_time_features(trips_df)

    for prefix in ["hour", "dow", "month"]:
        sin_col = f"{prefix}_sin"
        cos_col = f"{prefix}_cos"
        assert sin_col in df_cyc.columns and cos_col in df_cyc.columns

        norm = df_cyc[sin_col] ** 2 + df_cyc[cos_col] ** 2
        assert np.allclose(norm, 1.0, atol=1e-5), f"Cyclical norm for {prefix} violates sin^2 + cos^2 = 1"
        assert (df_cyc[sin_col] >= -1.0).all() and (df_cyc[sin_col] <= 1.0).all()
        assert (df_cyc[cos_col] >= -1.0).all() and (df_cyc[cos_col] <= 1.0).all()


def test_historical_lookup_isolated_to_train(trips_df: pd.DataFrame):
    """Verify that historical lookup table accurately aggregates training statistics with zero test leakage."""
    train_df, _, _ = split_data_chronological(trips_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    lookup = build_historical_lag_lookup(train_df)

    assert "route_dow_hour_stats" in lookup
    assert "route_stats" in lookup
    assert "global_median" in lookup

    # Lookup max date must be <= max train date
    max_train_date = str(train_df["departure_datetime"].max())
    assert lookup["training_max_date"] == max_train_date

    # Verify that all routes in train are in the lookup
    for r in train_df["route_id"].unique():
        assert r in lookup["route_stats"]
