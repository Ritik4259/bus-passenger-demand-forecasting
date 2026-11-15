"""Feature engineering module with strict chronological splitting and 24h leakage-safe lag extraction.

Ensures zero temporal data leakage:
1. Strict chronological train / validation / test splits by datetime.
2. Continuous cyclical encodings for diurnal, weekly, and annual cycles.
3. Strict T - 24h cutoff for all historical lag and rolling statistics.
4. Exportable historical lag lookup table computed strictly on training data for self-contained inference.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def split_data_chronological(
    df: pd.DataFrame,
    datetime_col: str = "departure_datetime",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train, validation, and test subsets.

    Args:
        df: Input DataFrame containing transit trips.
        datetime_col: Column name representing trip departure timestamp.
        train_ratio: Proportion of timeline allocated to training (default 0.70).
        val_ratio: Proportion of timeline allocated to validation (default 0.15).
        test_ratio: Proportion of timeline allocated to test (default 0.15).

    Returns:
        Tuple of (train_df, val_df, test_df) strictly non-overlapping in time.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    df_sorted = df.copy()
    df_sorted[datetime_col] = pd.to_datetime(df_sorted[datetime_col])
    df_sorted = df_sorted.sort_values(by=[datetime_col, "route_id"]).reset_index(drop=True)

    min_date = df_sorted[datetime_col].min()
    max_date = df_sorted[datetime_col].max()
    total_duration = max_date - min_date

    train_cutoff = min_date + total_duration * train_ratio
    val_cutoff = min_date + total_duration * (train_ratio + val_ratio)

    train_df = df_sorted[df_sorted[datetime_col] < train_cutoff].copy().reset_index(drop=True)
    val_df = (
        df_sorted[
            (df_sorted[datetime_col] >= train_cutoff) & (df_sorted[datetime_col] < val_cutoff)
        ]
        .copy()
        .reset_index(drop=True)
    )
    test_df = df_sorted[df_sorted[datetime_col] >= val_cutoff].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trigonometric cyclical representations of hour, day-of-week, and month.

    Preserves the continuity between hour 23 and hour 0, Sunday and Monday, Dec and Jan.
    """
    df = df.copy()
    hour = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # day_of_week is 0-6 (Mon-Sun)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)

    # month is 1-12
    df["month_sin"] = np.sin(2.0 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (df["month"] - 1) / 12.0)

    return df


def add_strict_24h_lag_features(
    df: pd.DataFrame,
    datetime_col: str = "departure_datetime",
    demand_col: str = "passenger_demand",
) -> pd.DataFrame:
    """Compute historical lag features strictly respecting the T - 24h cutoff.

    Calculates:
    - `lag_24h_demand`: Demand from the most recent trip on the same route at or before T - 24h.
    - `lag_7d_demand`: Demand from the trip on the same route at or before T - 7 days.
    - `rolling_7d_mean`: 7-day rolling mean demand for the route computed strictly from data <= T - 24h.
    - `lag_time_24h`: Actual timestamp of the matched 24h lag trip (used for cryptographic leakage verification).
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=[datetime_col, "route_id"]).reset_index(drop=True)

    # 1. 24-hour lag via backward merge_asof on (T - 24h)
    df["cutoff_24h"] = df[datetime_col] - pd.Timedelta(hours=24)
    lookup_24h = (
        df[[datetime_col, "route_id", demand_col]]
        .rename(columns={datetime_col: "lag_time_24h", demand_col: "lag_24h_demand"})
        .sort_values(by="lag_time_24h")
    )
    df = pd.merge_asof(
        df,
        lookup_24h,
        left_on="cutoff_24h",
        right_on="lag_time_24h",
        by="route_id",
        direction="backward",
    )

    # 2. 7-day lag via backward merge_asof on (T - 7 days)
    df["cutoff_7d"] = df[datetime_col] - pd.Timedelta(days=7)
    lookup_7d = (
        df[[datetime_col, "route_id", demand_col]]
        .rename(columns={datetime_col: "lag_time_7d", demand_col: "lag_7d_demand"})
        .sort_values(by="lag_time_7d")
    )
    df = pd.merge_asof(
        df,
        lookup_7d,
        left_on="cutoff_7d",
        right_on="lag_time_7d",
        by="route_id",
        direction="backward",
    )

    # 3. 7-day rolling mean demand strictly shifted by 24h
    rolling_records = []
    for route_id, route_group in df.groupby("route_id"):
        route_group = route_group.sort_values(by=datetime_col).copy()
        # Compute 7-day rolling mean on historical departures
        rolling_series = (
            route_group.set_index(datetime_col)[demand_col]
            .rolling("7D", min_periods=3)
            .mean()
        )
        route_group["rolling_7d_raw"] = rolling_series.values
        rolling_records.append(
            route_group[[datetime_col, "route_id", "rolling_7d_raw"]]
        )

    rolling_lookup = (
        pd.concat(rolling_records)
        .sort_values(by=datetime_col)
        .rename(columns={datetime_col: "rolling_time", "rolling_7d_raw": "rolling_7d_mean"})
    )
    df = pd.merge_asof(
        df,
        rolling_lookup,
        left_on="cutoff_24h",
        right_on="rolling_time",
        by="route_id",
        direction="backward",
    )

    # Drop temporary calculation columns (keep lag_time_24h for leakage verification tests)
    df = df.drop(columns=["cutoff_24h", "cutoff_7d", "lag_time_7d", "rolling_time"])
    return df


def build_historical_lag_lookup(train_df: pd.DataFrame) -> Dict[str, Any]:
    """Construct historical summary and lag fallback dictionary strictly from the training partition.

    This lookup is serialized alongside the trained model so that online inference can:
    1. Retrieve route baseline statistics without database calls.
    2. Fall back to median/mean values if lag features are absent in a raw inference query.
    """
    train_clean = train_df.dropna(subset=["passenger_demand"]).copy()

    # Route + DayOfWeek + Hour statistics
    route_dow_hour_stats = (
        train_clean.groupby(["route_id", "day_of_week", "hour"])["passenger_demand"]
        .agg(["median", "mean", "std"])
        .reset_index()
    )
    route_dow_hour_dict = {}
    for _, row in route_dow_hour_stats.iterrows():
        key = f"{row['route_id']}_DOW{int(row['day_of_week'])}_H{int(row['hour'])}"
        route_dow_hour_dict[key] = {
            "median": float(row["median"]),
            "mean": float(row["mean"]),
            "std": float(row["std"]) if pd.notna(row["std"]) else 0.0,
        }

    # Route + Hour fallback
    route_hour_stats = (
        train_clean.groupby(["route_id", "hour"])["passenger_demand"]
        .agg(["median", "mean"])
        .reset_index()
    )
    route_hour_dict = {}
    for _, row in route_hour_stats.iterrows():
        key = f"{row['route_id']}_H{int(row['hour'])}"
        route_hour_dict[key] = {
            "median": float(row["median"]),
            "mean": float(row["mean"]),
        }

    # Global route-level statistics
    route_stats = (
        train_clean.groupby("route_id")["passenger_demand"]
        .agg(["median", "mean"])
        .reset_index()
    )
    route_dict = {
        row["route_id"]: {
            "median": float(row["median"]),
            "mean": float(row["mean"]),
        }
        for _, row in route_stats.iterrows()
    }

    # Overall dataset median
    global_median = float(train_clean["passenger_demand"].median())

    return {
        "route_dow_hour_stats": route_dow_hour_dict,
        "route_hour_stats": route_hour_dict,
        "route_stats": route_dict,
        "global_median": global_median,
        "training_min_date": str(train_clean["departure_datetime"].min()),
        "training_max_date": str(train_clean["departure_datetime"].max()),
    }


def prepare_features(
    df: pd.DataFrame,
    lag_lookup: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """End-to-end feature preparation: cyclical time transforms, 24h lags, and fallback imputation."""
    df_feat = add_cyclical_time_features(df)
    df_feat = add_strict_24h_lag_features(df_feat)

    # Impute missing initial lags (first 24h/7d) using historical lookup if provided
    if lag_lookup:
        global_med = lag_lookup["global_median"]
        df_feat["lag_24h_demand"] = df_feat["lag_24h_demand"].fillna(global_med)
        df_feat["lag_7d_demand"] = df_feat["lag_7d_demand"].fillna(global_med)
        df_feat["rolling_7d_mean"] = df_feat["rolling_7d_mean"].fillna(global_med)

    return df_feat


# Canonical feature column names consumed by ML models (capacity is deliberately excluded to prevent reverse causality)
FEATURE_COLUMNS = [
    "route_id",
    "is_weekend",
    "day_of_week",
    "hour",
    "month",
    "is_raining",
    "temperature_c",
    "is_special_event",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "lag_24h_demand",
    "lag_7d_demand",
    "rolling_7d_mean",
]

TARGET_COLUMN = "passenger_demand"
