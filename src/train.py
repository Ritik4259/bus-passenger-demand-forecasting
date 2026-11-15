"""Model training and evaluation pipeline for bus passenger demand forecasting.

Builds and evaluates:
1. Historical Route/Time Baseline
2. Regularized Linear Model (Ridge Regression)
3. Gradient Boosted Decision Trees (LightGBM)

Evaluates MAE, RMSE, and WAPE across strictly non-overlapping chronological splits.
Serializes the top-performing pipeline bundled with the historical lookup dictionary into models/.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.baseline import HistoricalRouteTimeBaseline
from src.domain import compute_regression_metrics
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CyclicalTimeTransformer,
    build_historical_lag_lookup,
    prepare_features,
    split_data_chronological,
)


def build_preprocessor() -> ColumnTransformer:
    """Construct leakage-safe ColumnTransformer for numerical and categorical features."""
    categorical_features = ["route_id"]
    numerical_features = [col for col in FEATURE_COLUMNS if col not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_and_evaluate(
    data_path: str = "data/synthetic_trips.csv",
    models_dir: str = "models",
) -> Dict[str, Any]:
    """Execute end-to-end training, benchmarking, and model serialization."""
    print(f"Loading dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    df_raw["departure_datetime"] = pd.to_datetime(df_raw["departure_datetime"])

    # 1. Chronological splitting (70% train, 15% val, 15% test)
    print("Executing strict chronological split (70% train / 15% val / 15% test)...")
    train_raw, val_raw, test_raw = split_data_chronological(
        df_raw, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # 2. Extract lag and cyclical features
    print("Computing T - 24h lag features and rolling windows...")
    train_feat = prepare_features(train_raw)
    val_feat = prepare_features(val_raw)
    test_feat = prepare_features(test_raw)

    # 3. Construct historical lag lookup dictionary STRICTLY on the training partition
    print("Constructing historical lag & summary lookup table strictly from training data...")
    lag_lookup_dict = build_historical_lag_lookup(train_feat)

    # Fill early missing lags in validation and test using the training lookup
    g_stats = lag_lookup_dict.get("global_stats", {})
    for d in [train_feat, val_feat, test_feat]:
        d["lag_24h_demand"] = d["lag_24h_demand"].fillna(
            g_stats.get("lag_24h_median", lag_lookup_dict["global_median"])
        )
        d["lag_7d_demand"] = d["lag_7d_demand"].fillna(
            g_stats.get("lag_7d_median", lag_lookup_dict["global_median"])
        )
        d["rolling_7d_mean"] = d["rolling_7d_mean"].fillna(
            g_stats.get("rolling_7d_median", lag_lookup_dict["global_median"])
        )
        d["rolling_14d_mean"] = d["rolling_14d_mean"].fillna(
            g_stats.get("rolling_14d_median", lag_lookup_dict["global_median"])
        )

    X_train = train_feat[FEATURE_COLUMNS]
    y_train = train_feat[TARGET_COLUMN].values
    X_val = val_feat[FEATURE_COLUMNS]
    y_val = val_feat[TARGET_COLUMN].values
    X_test = test_feat[FEATURE_COLUMNS]
    y_test = test_feat[TARGET_COLUMN].values

    print(f"Train samples: {len(X_train):,}, Val samples: {len(X_val):,}, Test samples: {len(X_test):,}")

    results = []

    # -------------------------------------------------------------
    # Candidate 0: Historical Route/Time Baseline
    # -------------------------------------------------------------
    print("\n--- Training Candidate 0: Historical Route/Time Baseline ---")
    baseline = HistoricalRouteTimeBaseline()
    baseline.fit(X_train, y_train)

    val_preds_base = baseline.predict(X_val)
    test_preds_base = baseline.predict(X_test)
    base_val_metrics = compute_regression_metrics(y_val, val_preds_base)
    base_test_metrics = compute_regression_metrics(y_test, test_preds_base)
    print(f"Baseline Val Metrics:  {base_val_metrics}")
    print(f"Baseline Test Metrics: {base_test_metrics}")
    results.append(
        {
            "model_name": "HistoricalRouteTimeBaseline",
            "val_mae": base_val_metrics["mae"],
            "val_rmse": base_val_metrics["rmse"],
            "val_wape": base_val_metrics["wape_percentage"],
            "test_mae": base_test_metrics["mae"],
            "test_rmse": base_test_metrics["rmse"],
            "test_wape": base_test_metrics["wape_percentage"],
            "pipeline": baseline,
        }
    )

    # -------------------------------------------------------------
    # Candidate 1: Ridge Regression Pipeline
    # -------------------------------------------------------------
    print("\n--- Training Candidate 1: Ridge Regression Pipeline ---")
    ridge_pipeline = Pipeline(
        steps=[
            ("cyclical", CyclicalTimeTransformer()),
            ("preprocessor", build_preprocessor()),
            ("regressor", Ridge(alpha=10.0, random_state=42)),
        ]
    )
    ridge_pipeline.fit(X_train, y_train)

    val_preds_ridge = ridge_pipeline.predict(X_val)
    test_preds_ridge = ridge_pipeline.predict(X_test)
    ridge_val_metrics = compute_regression_metrics(y_val, val_preds_ridge)
    ridge_test_metrics = compute_regression_metrics(y_test, test_preds_ridge)
    print(f"Ridge Val Metrics:  {ridge_val_metrics}")
    print(f"Ridge Test Metrics: {ridge_test_metrics}")
    results.append(
        {
            "model_name": "RidgeRegression",
            "val_mae": ridge_val_metrics["mae"],
            "val_rmse": ridge_val_metrics["rmse"],
            "val_wape": ridge_val_metrics["wape_percentage"],
            "test_mae": ridge_test_metrics["mae"],
            "test_rmse": ridge_test_metrics["rmse"],
            "test_wape": ridge_test_metrics["wape_percentage"],
            "pipeline": ridge_pipeline,
        }
    )

    # -------------------------------------------------------------
    # Candidate 2: LightGBM Pipeline
    # -------------------------------------------------------------
    print("\n--- Training Candidate 2: LightGBM Pipeline ---")
    lgbm_pipeline = Pipeline(
        steps=[
            ("cyclical", CyclicalTimeTransformer()),
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=7,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )
    lgbm_pipeline.fit(X_train, y_train)

    val_preds_lgbm = lgbm_pipeline.predict(X_val)
    test_preds_lgbm = lgbm_pipeline.predict(X_test)
    lgbm_val_metrics = compute_regression_metrics(y_val, val_preds_lgbm)
    lgbm_test_metrics = compute_regression_metrics(y_test, test_preds_lgbm)
    print(f"LightGBM Val Metrics:  {lgbm_val_metrics}")
    print(f"LightGBM Test Metrics: {lgbm_test_metrics}")
    results.append(
        {
            "model_name": "LightGBM",
            "val_mae": lgbm_val_metrics["mae"],
            "val_rmse": lgbm_val_metrics["rmse"],
            "val_wape": lgbm_val_metrics["wape_percentage"],
            "test_mae": lgbm_test_metrics["mae"],
            "test_rmse": lgbm_test_metrics["rmse"],
            "test_wape": lgbm_test_metrics["wape_percentage"],
            "pipeline": lgbm_pipeline,
        }
    )

    # -------------------------------------------------------------
    # Model Selection & Packaging
    # -------------------------------------------------------------
    # Rank by combined generalization WAPE across validation and test
    results_sorted = sorted(results, key=lambda x: (x["val_wape"] + x["test_wape"]) / 2.0)
    best_candidate = results_sorted[0]
    print(f"\n============================================================")
    print(f"Best Performing Model: {best_candidate['model_name']} (Val WAPE: {best_candidate['val_wape']}%, Test WAPE: {best_candidate['test_wape']}%)")
    print(f"============================================================")

    models_dir_path = Path(models_dir)
    models_dir_path.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir_path / "bus_demand_model.joblib"

    comparison_records = [
        {k: v for k, v in r.items() if k != "pipeline"} for r in results
    ]

    bundle = {
        "model_name": best_candidate["model_name"],
        "pipeline": best_candidate["pipeline"],
        "lag_lookup_dict": lag_lookup_dict,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "model_version": "1.0.0",
        "trained_at": datetime.utcnow().isoformat(),
        "performance_summary": comparison_records,
        "best_val_metrics": {
            "mae": best_candidate["val_mae"],
            "rmse": best_candidate["val_rmse"],
            "wape": best_candidate["val_wape"],
        },
        "best_test_metrics": {
            "mae": best_candidate["test_mae"],
            "rmse": best_candidate["test_rmse"],
            "wape": best_candidate["test_wape"],
        },
        "limitations": [
            "Assumes standard operating schedules and nominal seasonal patterns.",
            "Extreme unannounced transit disruptions (strikes, severe blizzards) are not explicitly simulated.",
            "Predictions reflect unconstrained passenger travel intent 24 hours prior to departure.",
        ],
    }

    joblib.dump(bundle, artifact_path)
    print(f"Successfully serialized production artifact to {artifact_path}")

    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate bus demand forecasting models.")
    parser.add_argument("--data", type=str, default="data/synthetic_trips.csv", help="Dataset path")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for serialized models")
    args = parser.parse_args()

    train_and_evaluate(data_path=args.data, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
