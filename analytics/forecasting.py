"""Forecasting helpers for trend projection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from analytics.statistics import build_time_index


class ForecastingService:
    """Provide simple production-friendly forecasting primitives."""

    def project_linear_trend(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        periods: int = 12,
    ) -> dict[str, Any]:
        """Project the next periods of a numeric target with linear regression."""

        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' is not available.")

        working_df = dataframe.copy()
        target_series = pd.to_numeric(working_df[target_column], errors="coerce")
        time_index = build_time_index(working_df)

        if time_index is not None:
            series_df = pd.DataFrame({"period": time_index, target_column: target_series}).dropna()
            if series_df.empty:
                raise ValueError("Not enough valid time series rows for forecasting.")
            series_df = series_df.groupby("period", as_index=False)[target_column].sum()
            history_x = np.arange(len(series_df)).reshape(-1, 1)
            history_y = series_df[target_column].to_numpy()
            model = LinearRegression()
            model.fit(history_x, history_y)
            future_x = np.arange(len(series_df), len(series_df) + periods).reshape(-1, 1)
            forecast_values = model.predict(future_x)
            if len(series_df) > 1:
                deltas = series_df["period"].sort_values().diff().dropna()
                median_days = float(deltas.dt.days.median()) if not deltas.empty else 30.0
                frequency = "YS" if median_days >= 200 else "MS"
            else:
                frequency = "MS"
            offset = pd.offsets.YearBegin(1) if frequency == "YS" else pd.offsets.MonthBegin(1)
            future_periods = pd.date_range(
                start=series_df["period"].iloc[-1] + offset,
                periods=periods,
                freq=frequency,
            )
            forecast_df = pd.DataFrame(
                {
                    "period": future_periods,
                    target_column: np.round(forecast_values, 4),
                    "series": "forecast",
                }
            )
            history_df = series_df.copy()
            history_df["series"] = "history"
        else:
            numeric_series = target_series.dropna().reset_index(drop=True)
            if len(numeric_series) < 2:
                raise ValueError("Not enough numeric observations for forecasting.")
            history_x = np.arange(len(numeric_series)).reshape(-1, 1)
            history_y = numeric_series.to_numpy()
            model = LinearRegression()
            model.fit(history_x, history_y)
            future_x = np.arange(len(numeric_series), len(numeric_series) + periods).reshape(-1, 1)
            forecast_values = model.predict(future_x)
            history_df = pd.DataFrame(
                {
                    "period": np.arange(len(numeric_series)),
                    target_column: numeric_series,
                    "series": "history",
                }
            )
            forecast_df = pd.DataFrame(
                {
                    "period": np.arange(len(numeric_series), len(numeric_series) + periods),
                    target_column: np.round(forecast_values, 4),
                    "series": "forecast",
                }
            )

        combined = pd.concat([history_df, forecast_df], ignore_index=True)
        r2_score = float(model.score(history_x, history_y)) if len(history_y) > 1 else 0.0

        return {
            "model_type": "linear_trend_projection",
            "target_column": target_column,
            "periods": periods,
            "r2_score": round(r2_score, 4),
            "history": history_df,
            "forecast": forecast_df,
            "combined": combined,
        }
