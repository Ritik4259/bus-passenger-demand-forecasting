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

    def _extract_cols(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Safely extract (route_id, day_of_week, hour) arrays whether input is DataFrame or ndarray."""
        if isinstance(X, pd.DataFrame):
            hour_col = (
                self.hour_col
                if self.hour_col in X.columns
                else ("departure_hour" if "departure_hour" in X.columns else self.hour_col)
            )
            routes = X[self.route_col].astype(str).values
            dows = X[self.dow_col].astype(int).values
            hours = X[hour_col].astype(int).values
            return routes, dows, hours

        arr = np.asarray(X)
        if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
            names = list(self.feature_names_in_)
            route_idx = names.index(self.route_col) if self.route_col in names else 0
            dow_idx = names.index(self.dow_col) if self.dow_col in names else 1
            hour_col_name = (
                self.hour_col
                if self.hour_col in names
                else ("departure_hour" if "departure_hour" in names else None)
            )
            hour_idx = names.index(hour_col_name) if hour_col_name in names else 2
        else:
            if arr.shape[1] == 3:
                route_idx, dow_idx, hour_idx = 0, 1, 2
            elif arr.shape[1] >= 4:
                # Default FEATURE_COLUMNS order: route_id (0), is_weekend (1), day_of_week (2), hour (3)
                route_idx, dow_idx, hour_idx = 0, 2, 3
            else:
                route_idx, dow_idx, hour_idx = 0, 1, 2

        routes = arr[:, route_idx].astype(str)
        dows = arr[:, dow_idx].astype(int)
        hours = arr[:, hour_idx].astype(int)
        return routes, dows, hours

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
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError("Training data y cannot be empty.")

        routes, dows, hours = self._extract_cols(X)

        df_fit = pd.DataFrame(
            {
                "route_id": routes,
                "day_of_week": dows,
                "hour": hours,
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

        routes, dows, hours = self._extract_cols(X)

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
