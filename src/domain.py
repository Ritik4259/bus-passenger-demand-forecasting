"""Domain models, canonical action flag logic, and shared evaluation metrics."""

from enum import Enum
from typing import Any, Dict, Tuple, Union

import numpy as np


class ActionFlag(str, Enum):
    """Canonical operational action flags for fleet sizing and dispatch."""

    UNDERUTILIZED = "UNDERUTILIZED"
    OPTIMAL = "OPTIMAL"
    OVERFLOW_RISK = "OVERFLOW_RISK"


def compute_action_flag(
    predicted_demand: float,
    capacity: int,
) -> Tuple[ActionFlag, str]:
    """Calculate operational action flag and recommendation from predicted demand and bus capacity.

    Args:
        predicted_demand: Predicted unconstrained passenger demand.
        capacity: Assigned vehicle seating/standing capacity.

    Returns:
        Tuple of (ActionFlag, operational_recommendation_text).
    """
    if capacity <= 0:
        raise ValueError("Capacity must be positive.")

    occupancy_rate = predicted_demand / float(capacity)

    if occupancy_rate < 0.50:
        flag = ActionFlag.UNDERUTILIZED
        recommendation = (
            "Low occupancy forecasted. Consider assigning a smaller vehicle (e.g. minibus) "
            "or consolidating scheduled trips to minimize operational expenditure."
        )
    elif occupancy_rate <= 0.90:
        flag = ActionFlag.OPTIMAL
        recommendation = (
            "Balanced passenger load forecasted. Scheduled bus capacity matches expected demand."
        )
    else:
        flag = ActionFlag.OVERFLOW_RISK
        recommendation = (
            "High risk of passenger overflow and denied boardings. Dispatch supplemental "
            "relief bus or upsize vehicle to high-capacity articulated fleet."
        )

    return flag, recommendation


def compute_regression_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, Any]:
    """Calculate MAE, RMSE, and WAPE across actual and predicted demand vectors.

    Args:
        y_true: True unconstrained passenger demand.
        y_pred: Model predicted passenger demand (clipped at 0).

    Returns:
        Dict containing sample_count, mae, rmse, wape, and wape_percentage.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.maximum(0.0, np.asarray(y_pred, dtype=float))

    if len(y_true_arr) == 0:
        return {
            "sample_count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "wape": 0.0,
            "wape_percentage": 0.0,
        }

    diff = np.abs(y_true_arr - y_pred_arr)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    sum_true = float(np.sum(y_true_arr))
    wape = float(np.sum(diff) / sum_true) if sum_true > 0 else 0.0

    return {
        "sample_count": int(len(y_true_arr)),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "wape": round(wape, 4),
        "wape_percentage": round(wape * 100.0, 2),
    }
