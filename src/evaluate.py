"""Evaluation and diagnostics module for bus passenger demand forecasting.

Computes:
1. Overall test performance (MAE, RMSE, WAPE).
2. Per-route breakdown of error metrics.
3. Peak vs. off-peak performance breakdown (Peak: 7-10 AM and 5-8 PM).
4. Capacity-aware action flag classification diagnostics (confusion matrix & overflow recall).
5. Exports structured JSON report to artifacts/evaluation_report.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.domain import ActionFlag, compute_action_flag, compute_regression_metrics
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_features,
    split_data_chronological,
)


def run_evaluation(
    data_path: str = "data/synthetic_trips.csv",
    model_path: str = "models/bus_demand_model.joblib",
    output_report_path: str = "artifacts/evaluation_report.json",
) -> Dict[str, Any]:
    """Execute comprehensive model evaluation on the chronological test split."""
    print(f"Loading model artifact from {model_path}...")
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    model_name = artifact.get("model_name", "UnknownModel")
    lag_lookup = artifact["lag_lookup_dict"]

    print(f"Loading test split from {data_path}...")
    df_raw = pd.read_csv(data_path)
    df_raw["departure_datetime"] = pd.to_datetime(df_raw["departure_datetime"])

    # Strict chronological split - evaluate exclusively on test partition
    _, _, test_raw = split_data_chronological(df_raw, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    test_feat = prepare_features(test_raw, lag_lookup=lag_lookup)

    X_test = test_feat[FEATURE_COLUMNS]
    y_true = test_feat[TARGET_COLUMN].values

    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_pred = np.maximum(0.0, y_pred)
    test_feat["predicted_demand"] = y_pred

    # 1. Overall Metrics
    overall_metrics = compute_regression_metrics(y_true, y_pred)
    print("\n========================================================")
    print(f"OVERALL TEST EVALUATION ({model_name})")
    print("========================================================")
    print(f"Test Samples: {overall_metrics['sample_count']:,}")
    print(f"MAE:  {overall_metrics['mae']}")
    print(f"RMSE: {overall_metrics['rmse']}")
    print(f"WAPE: {overall_metrics['wape_percentage']}%")

    # 2. Peak vs. Off-Peak Breakdown
    # Peak definition: 7:00 - 10:59 AM and 17:00 - 20:59 PM (7-10 and 17-20 inclusive)
    test_feat["is_peak_hour"] = test_feat["hour"].isin([7, 8, 9, 10, 17, 18, 19, 20])

    peak_df = test_feat[test_feat["is_peak_hour"]]
    off_peak_df = test_feat[~test_feat["is_peak_hour"]]

    peak_metrics = compute_regression_metrics(
        peak_df[TARGET_COLUMN].values, peak_df["predicted_demand"].values
    )
    off_peak_metrics = compute_regression_metrics(
        off_peak_df[TARGET_COLUMN].values, off_peak_df["predicted_demand"].values
    )

    print("\n--------------------------------------------------------")
    print("PEAK VS. OFF-PEAK BREAKDOWN (Peak: 7-10 AM & 5-8 PM)")
    print("--------------------------------------------------------")
    print(
        f"Peak     | Samples: {peak_metrics['sample_count']:5d} | MAE: {peak_metrics['mae']:6.2f} | RMSE: {peak_metrics['rmse']:6.2f} | WAPE: {peak_metrics['wape_percentage']:5.2f}%"
    )
    print(
        f"Off-Peak | Samples: {off_peak_metrics['sample_count']:5d} | MAE: {off_peak_metrics['mae']:6.2f} | RMSE: {off_peak_metrics['rmse']:6.2f} | WAPE: {off_peak_metrics['wape_percentage']:5.2f}%"
    )

    # 3. Per-Route Breakdown
    per_route_metrics = {}
    print("\n--------------------------------------------------------")
    print("PER-ROUTE EVALUATION BREAKDOWN")
    print("--------------------------------------------------------")
    for route_id, group in test_feat.groupby("route_id"):
        route_name = group["route_name"].iloc[0] if "route_name" in group.columns else route_id
        r_metrics = compute_regression_metrics(
            group[TARGET_COLUMN].values, group["predicted_demand"].values
        )
        r_metrics["route_name"] = route_name
        r_metrics["capacity"] = int(group["capacity"].iloc[0])
        per_route_metrics[route_id] = r_metrics
        print(
            f"{route_id:10s} ({route_name:25s}) | Samples: {r_metrics['sample_count']:4d} | MAE: {r_metrics['mae']:5.2f} | RMSE: {r_metrics['rmse']:5.2f} | WAPE: {r_metrics['wape_percentage']:5.2f}%"
        )

    # 4. Action Flag Decision Diagnostics via shared domain logic
    test_feat["true_action_flag"] = [
        compute_action_flag(d, c)[0].value
        for d, c in zip(test_feat[TARGET_COLUMN], test_feat["capacity"])
    ]
    test_feat["pred_action_flag"] = [
        compute_action_flag(d, c)[0].value
        for d, c in zip(test_feat["predicted_demand"], test_feat["capacity"])
    ]

    flag_acc = float(accuracy_score(test_feat["true_action_flag"], test_feat["pred_action_flag"]))
    flag_report = classification_report(
        test_feat["true_action_flag"], test_feat["pred_action_flag"], output_dict=True
    )
    overflow_recall = float(flag_report.get("OVERFLOW_RISK", {}).get("recall", 0.0))

    print("\n--------------------------------------------------------")
    print("OPERATIONAL ACTION FLAG PERFORMANCE")
    print("--------------------------------------------------------")
    print(f"Overall Action Flag Accuracy: {flag_acc * 100:.2f}%")
    print(f"OVERFLOW_RISK Recall:         {overflow_recall * 100:.2f}%")

    report = {
        "model_name": model_name,
        "evaluation_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "test_dataset_size": len(test_feat),
        "overall_metrics": overall_metrics,
        "temporal_breakdown": {
            "peak_hours": peak_metrics,
            "off_peak_hours": off_peak_metrics,
        },
        "per_route_breakdown": per_route_metrics,
        "action_flag_performance": {
            "overall_accuracy": round(flag_acc, 4),
            "overflow_risk_recall": round(overflow_recall, 4),
            "classification_details": flag_report,
        },
        "model_limitations": artifact.get("limitations", []),
    }

    report_path = Path(output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation report successfully saved to: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate bus demand forecasting model.")
    parser.add_argument("--data", type=str, default="data/synthetic_trips.csv", help="Dataset path")
    parser.add_argument("--model", type=str, default="models/bus_demand_model.joblib", help="Model path")
    parser.add_argument(
        "--output", type=str, default="artifacts/evaluation_report.json", help="Report output path"
    )
    args = parser.parse_args()

    run_evaluation(data_path=args.data, model_path=args.model, output_report_path=args.output)


if __name__ == "__main__":
    main()
