"""Data containers for machine learning outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ForecastOutput:
    """Output for a time series forecast."""

    target_column: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    combined: pd.DataFrame
    r2_score: float
    model_type: str


@dataclass(slots=True)
class ClusterOutput:
    """Output for clustering analysis."""

    annotated_frame: pd.DataFrame
    feature_columns: list[str]
    cluster_count: int


@dataclass(slots=True)
class AnomalyOutput:
    """Output for anomaly detection."""

    annotated_frame: pd.DataFrame
    feature_columns: list[str]
    anomaly_count: int
