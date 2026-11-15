"""Historical Route/Time Baseline Estimator.

Scikit-learn compatible baseline that predicts median passenger demand grouped by
(route_id, day_of_week, departure_hour) computed strictly from the training partition.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class HistoricalRouteTimeBaseline(RegressorMixin, BaseEstimator):
    """Benchmark baseline estimating demand via historical route-by-time median.

    Implements hierarchical fallback to ensure robustness against rare/unseen trips:
    1. Exact match: (route_id, day_of_week, hour)
    2. Route-hour match: (route_id, hour)
    3. Route match: (route_id)
    4. Global dataset median
    """

    def __init__(
        self,
        aggregation: str = "median",
        route_col: str = "route_id",
        dow_col: str = "day_of_week",
        hour_col: str = "hour",
    ):
        self.aggregation = aggregation
        self.route_col = route_col
        self.dow_col = dow_col
        self.hour_col = hour_col

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "HistoricalRouteTimeBaseline":
        """Fit the baseline strictly on the training partition.

        Args:
            X: Training features containing route_id, day_of_week, and hour.
            y: Target passenger demand series.

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            df_x = X.copy()
        else:
            # Assume columns: [route_id, ..., day_of_week, hour]
            df_x = pd.DataFrame(X)
            # Default to first 3 columns or mapped names
            if 0 in df_x.columns and 1 in df_x.columns and 2 in df_x.columns:
                df_x = df_x.rename(
                    columns={0: self.route_col, 1: self.dow_col, 2: self.hour_col}
                )

        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError("Training data y cannot be empty.")

        df_fit = pd.DataFrame(
            {
                "route_id": df_x[self.route_col].astype(str).values,
                "day_of_week": df_x[self.dow_col].astype(int).values,
                "hour": df_x[self.hour_col].astype(int).values,
                "target": y_arr,
            }
        )

        agg_func = "median" if self.aggregation == "median" else "mean"

        # 1. Primary: (route_id, day_of_week, hour)
        primary_agg = (
            df_fit.groupby(["route_id", "day_of_week", "hour"])["target"]
            .agg(agg_func)
            .to_dict()
        )
        self.group_medians_: Dict[Tuple[str, int, int], float] = {
            k: float(v) for k, v in primary_agg.items()
        }

        # 2. Secondary fallback: (route_id, hour)
        sec_agg = df_fit.groupby(["route_id", "hour"])["target"].agg(agg_func).to_dict()
        self.route_hour_medians_: Dict[Tuple[str, int], float] = {
            k: float(v) for k, v in sec_agg.items()
        }

        # 3. Tertiary fallback: route_id
        route_agg = df_fit.groupby("route_id")["target"].agg(agg_func).to_dict()
        self.route_medians_: Dict[str, float] = {k: float(v) for k, v in route_agg.items()}

        # 4. Global fallback
        self.global_median_: float = (
            float(np.median(y_arr)) if self.aggregation == "median" else float(np.mean(y_arr))
        )

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict passenger demand for input trips using fitted training lookups."""
        check_is_fitted(self, attributes=["is_fitted_", "group_medians_"])

        if isinstance(X, pd.DataFrame):
            df_x = X
        else:
            df_x = pd.DataFrame(X)
            if 0 in df_x.columns and 1 in df_x.columns and 2 in df_x.columns:
                df_x = df_x.rename(
                    columns={0: self.route_col, 1: self.dow_col, 2: self.hour_col}
                )

        routes = df_x[self.route_col].astype(str).values
        dows = df_x[self.dow_col].astype(int).values
        hours = df_x[self.hour_col].astype(int).values

        preds = np.empty(len(routes), dtype=float)
        for i in range(len(routes)):
            r, d, h = routes[i], dows[i], hours[i]
            # Level 1: exact group match
            if (r, d, h) in self.group_medians_:
                preds[i] = self.group_medians_[(r, d, h)]
            # Level 2: route + hour
            elif (r, h) in self.route_hour_medians_:
                preds[i] = self.route_hour_medians_[(r, h)]
            # Level 3: route
            elif r in self.route_medians_:
                preds[i] = self.route_medians_[r]
            # Level 4: global dataset median
            else:
                preds[i] = self.global_median_

        return np.maximum(0.0, preds)
