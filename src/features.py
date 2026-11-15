"""Feature engineering module with strict chronological splitting and 24h leakage-safe lag extraction.

Ensures zero temporal data leakage:
1. Strict chronological train / validation / test splits by datetime.
2. Continuous cyclical encodings for diurnal, weekly, and annual cycles via CyclicalTimeTransformer.
3. Strict T - 24h cutoff for all historical lag and rolling statistics (including 14-day rolling mean).
4. Exportable historical lag lookup table computed strictly on training data for self-contained inference.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Indian / Punjab public & academic holiday calendar for 2026
PUNJAB_ACADEMIC_HOLIDAYS_2026 = {
    "2026-01-26",  # Republic Day
    "2026-02-15",  # Maha Shivratri
    "2026-03-03",  # Holi / Hola Mohalla
    "2026-03-21",  # Id-ul-Fitr
    "2026-03-27",  # Ram Navami
    "2026-03-31",  # Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Baisakhi / Dr. Ambedkar Jayanti
    "2026-05-01",  # May Day
    "2026-05-27",  # Id-ul-Zuha
    "2026-05-31",  # Buddha Purnima
    "2026-06-26",  # Muharram
    "2026-08-15",  # Independence Day
    "2026-09-04",  # Janmashtami
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-08",  # Diwali
    "2026-11-24",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
}


def is_punjab_holiday(dt: datetime) -> int:
    """Return 1 if date falls on a Punjab academic/public holiday, else 0."""
    return int(dt.strftime("%Y-%m-%d") in PUNJAB_ACADEMIC_HOLIDAYS_2026)


class CyclicalTimeTransformer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer generating continuous cyclical sin/cos encodings.

    Transforms hour, day_of_week, and month into unit-circle coordinates inside
    the serialized scikit-learn Pipeline (satisfying ADR-0003).
    """

    def __init__(
        self,
        hour_col: str = "hour",
        dow_col: str = "day_of_week",
        month_col: str = "month",
        minute_col: str = "minute",
    ):
        self.hour_col = hour_col
        self.dow_col = dow_col
        self.month_col = month_col
        self.minute_col = minute_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        if "hour_sin" not in df.columns or "hour_cos" not in df.columns:
            minute = df[self.minute_col] if self.minute_col in df.columns else 0.0
            hour = df[self.hour_col] + minute / 60.0
            df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
            df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

        if "dow_sin" not in df.columns or "dow_cos" not in df.columns:
            dow = df[self.dow_col]
            df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
            df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

        if "month_sin" not in df.columns or "month_cos" not in df.columns:
            month = df[self.month_col]
            df["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
            df["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)

        return df


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
    """Generate trigonometric cyclical representations of hour, day-of-week, and month."""
    transformer = CyclicalTimeTransformer()
    return transformer.transform(df)


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
    - `rolling_14d_mean`: 14-day rolling mean demand for the route computed strictly from data <= T - 24h.
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

    # 3. 7-day and 14-day rolling mean demand strictly shifted by 24h
    rolling_records = []
    for route_id, route_group in df.groupby("route_id"):
        route_group = route_group.sort_values(by=datetime_col).copy()
        rolling_7d = (
            route_group.set_index(datetime_col)[demand_col]
            .rolling("7D", min_periods=3)
            .mean()
        )
        rolling_14d = (
            route_group.set_index(datetime_col)[demand_col]
            .rolling("14D", min_periods=5)
            .mean()
        )
        route_group["rolling_7d_raw"] = rolling_7d.values
        route_group["rolling_14d_raw"] = rolling_14d.values
        rolling_records.append(
            route_group[[datetime_col, "route_id", "rolling_7d_raw", "rolling_14d_raw"]]
        )

    rolling_lookup = (
        pd.concat(rolling_records)
        .sort_values(by=datetime_col)
        .rename(
            columns={
                datetime_col: "rolling_time",
                "rolling_7d_raw": "rolling_7d_mean",
                "rolling_14d_raw": "rolling_14d_mean",
            }
        )
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

    Stores distinct calibrated medians for:
    - lag_24h_demand (prior-day same-hour demand)
    - lag_7d_demand (prior-week same-day demand)
    - rolling_7d_mean (7-day rolling baseline)
    - rolling_14d_mean (14-day rolling baseline)
    Eliminating train-serve lag skew at inference time.
    """
    train_clean = train_df.dropna(subset=["passenger_demand"]).copy()

    # Route + DayOfWeek + Hour statistics
    lag_cols = ["passenger_demand", "lag_24h_demand", "lag_7d_demand", "rolling_7d_mean", "rolling_14d_mean"]
    available_cols = [c for c in lag_cols if c in train_clean.columns]

    route_dow_hour_stats = (
        train_clean.groupby(["route_id", "day_of_week", "hour"])[available_cols]
        .median()
        .reset_index()
    )
    route_dow_hour_dict = {}
    for _, row in route_dow_hour_stats.iterrows():
        key = f"{row['route_id']}_DOW{int(row['day_of_week'])}_H{int(row['hour'])}"
        route_dow_hour_dict[key] = {
            "demand_median": float(row["passenger_demand"]),
            "lag_24h_median": float(row.get("lag_24h_demand", row["passenger_demand"])),
            "lag_7d_median": float(row.get("lag_7d_demand", row["passenger_demand"])),
            "rolling_7d_median": float(row.get("rolling_7d_mean", row["passenger_demand"])),
            "rolling_14d_median": float(row.get("rolling_14d_mean", row["passenger_demand"])),
        }

    # Route + Hour fallback
    route_hour_stats = (
        train_clean.groupby(["route_id", "hour"])[available_cols]
        .median()
        .reset_index()
    )
    route_hour_dict = {}
    for _, row in route_hour_stats.iterrows():
        key = f"{row['route_id']}_H{int(row['hour'])}"
        route_hour_dict[key] = {
            "demand_median": float(row["passenger_demand"]),
            "lag_24h_median": float(row.get("lag_24h_demand", row["passenger_demand"])),
            "lag_7d_median": float(row.get("lag_7d_demand", row["passenger_demand"])),
            "rolling_7d_median": float(row.get("rolling_7d_mean", row["passenger_demand"])),
            "rolling_14d_median": float(row.get("rolling_14d_mean", row["passenger_demand"])),
        }

    # Global route-level statistics
    route_stats = (
        train_clean.groupby("route_id")[available_cols]
        .median()
        .reset_index()
    )
    route_dict = {
        row["route_id"]: {
            "demand_median": float(row["passenger_demand"]),
            "lag_24h_median": float(row.get("lag_24h_demand", row["passenger_demand"])),
            "lag_7d_median": float(row.get("lag_7d_demand", row["passenger_demand"])),
            "rolling_7d_median": float(row.get("rolling_7d_mean", row["passenger_demand"])),
            "rolling_14d_median": float(row.get("rolling_14d_mean", row["passenger_demand"])),
        }
        for _, row in route_stats.iterrows()
    }

    # Global dataset medians
    global_stats = {
        "demand_median": float(train_clean["passenger_demand"].median()),
        "lag_24h_median": float(train_clean["lag_24h_demand"].median()) if "lag_24h_demand" in train_clean else float(train_clean["passenger_demand"].median()),
        "lag_7d_median": float(train_clean["lag_7d_demand"].median()) if "lag_7d_demand" in train_clean else float(train_clean["passenger_demand"].median()),
        "rolling_7d_median": float(train_clean["rolling_7d_mean"].median()) if "rolling_7d_mean" in train_clean else float(train_clean["passenger_demand"].median()),
        "rolling_14d_median": float(train_clean["rolling_14d_mean"].median()) if "rolling_14d_mean" in train_clean else float(train_clean["passenger_demand"].median()),
    }

    return {
        "route_dow_hour_stats": route_dow_hour_dict,
        "route_hour_stats": route_hour_dict,
        "route_stats": route_dict,
        "global_stats": global_stats,
        "global_median": global_stats["demand_median"],
        "training_min_date": str(train_clean["departure_datetime"].min()),
        "training_max_date": str(train_clean["departure_datetime"].max()),
    }


def prepare_features(
    df: pd.DataFrame,
    lag_lookup: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """End-to-end feature preparation: cyclical time transforms, 24h lags, and fallback imputation."""
    df_feat = df.copy()

    # Ensure is_holiday column exists
    if "is_holiday" not in df_feat.columns:
        dt_series = pd.to_datetime(df_feat["departure_datetime"])
        df_feat["is_holiday"] = dt_series.apply(is_punjab_holiday)

    df_feat = add_cyclical_time_features(df_feat)
    df_feat = add_strict_24h_lag_features(df_feat)

    # Impute missing initial lags (first 24h/7d) using historical lookup if provided
    if lag_lookup:
        g_stats = lag_lookup.get("global_stats", {})
        df_feat["lag_24h_demand"] = df_feat["lag_24h_demand"].fillna(g_stats.get("lag_24h_median", lag_lookup["global_median"]))
        df_feat["lag_7d_demand"] = df_feat["lag_7d_demand"].fillna(g_stats.get("lag_7d_median", lag_lookup["global_median"]))
        df_feat["rolling_7d_mean"] = df_feat["rolling_7d_mean"].fillna(g_stats.get("rolling_7d_median", lag_lookup["global_median"]))
        df_feat["rolling_14d_mean"] = df_feat["rolling_14d_mean"].fillna(g_stats.get("rolling_14d_median", lag_lookup["global_median"]))

    return df_feat


# Canonical feature column names consumed by ML models (capacity is deliberately excluded to prevent reverse causality)
FEATURE_COLUMNS = [
    "route_id",
    "is_weekend",
    "day_of_week",
    "hour",
    "month",
    "is_holiday",
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
    "rolling_14d_mean",
]

TARGET_COLUMN = "passenger_demand"
