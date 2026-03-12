"""Machine learning workflows used by the analyst platform."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from analytics.statistics import preferred_numeric_columns
from analytics.forecasting import ForecastingService
from ml.models import AnomalyOutput, ClusterOutput, ForecastOutput


class MLService:
    """Wrap classical ML models behind a consistent interface."""

    def __init__(self) -> None:
        self.forecasting_service = ForecastingService()

    def train_linear_regression(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> dict[str, Any]:
        """Fit a linear regression model on numeric features."""

        numeric_df = dataframe[preferred_numeric_columns(dataframe)].dropna()
        if target_column not in numeric_df.columns:
            raise ValueError(f"Target column '{target_column}' must be numeric.")

        feature_columns = [column for column in numeric_df.columns if column != target_column]
        if not feature_columns:
            raise ValueError("At least one numeric feature is required for regression.")

        X = numeric_df[feature_columns]
        y = numeric_df[target_column]
        model = LinearRegression()
        model.fit(X, y)
        score = model.score(X, y)
        return {
            "model_type": "linear_regression",
            "target_column": target_column,
            "feature_columns": feature_columns,
            "r2_score": round(float(score), 4),
            "coefficients": {
                column: round(float(coefficient), 6)
                for column, coefficient in zip(feature_columns, model.coef_, strict=False)
            },
            "intercept": round(float(model.intercept_), 6),
        }

    def time_series_projection(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        periods: int = 12,
    ) -> ForecastOutput:
        """Project a time series using a linear trend model."""

        payload = self.forecasting_service.project_linear_trend(
            dataframe=dataframe,
            target_column=target_column,
            periods=periods,
        )
        return ForecastOutput(
            target_column=payload["target_column"],
            history=payload["history"],
            forecast=payload["forecast"],
            combined=payload["combined"],
            r2_score=payload["r2_score"],
            model_type=payload["model_type"],
        )

    def cluster_dataset(
        self,
        dataframe: pd.DataFrame,
        n_clusters: int = 3,
    ) -> ClusterOutput:
        """Cluster numeric observations with K-Means."""

        numeric_columns = preferred_numeric_columns(dataframe)
        numeric_df = dataframe[numeric_columns].dropna()
        if numeric_df.shape[1] < 2:
            raise ValueError("Clustering requires at least two numeric columns.")

        feature_columns = list(numeric_df.columns[: min(4, numeric_df.shape[1])])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df[feature_columns])
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(scaled)
        annotated = dataframe.loc[numeric_df.index].copy()
        annotated["cluster"] = labels
        return ClusterOutput(
            annotated_frame=annotated,
            feature_columns=feature_columns,
            cluster_count=n_clusters,
        )

    def detect_anomalies(
        self,
        dataframe: pd.DataFrame,
        contamination: float = 0.05,
    ) -> AnomalyOutput:
        """Detect anomalies with Isolation Forest."""

        numeric_columns = preferred_numeric_columns(dataframe)
        numeric_df = dataframe[numeric_columns].dropna()
        if numeric_df.shape[1] < 2:
            raise ValueError("Anomaly detection requires at least two numeric columns.")

        feature_columns = list(numeric_df.columns[: min(4, numeric_df.shape[1])])
        model = IsolationForest(random_state=42, contamination=contamination)
        labels = model.fit_predict(numeric_df[feature_columns])
        annotated = dataframe.loc[numeric_df.index].copy()
        annotated["anomaly_flag"] = labels
        anomaly_count = int((labels == -1).sum())
        return AnomalyOutput(
            annotated_frame=annotated,
            feature_columns=feature_columns,
            anomaly_count=anomaly_count,
        )
