"""Agent responsible for chart generation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.config import AppConfig
from ml.models import AnomalyOutput, ClusterOutput, ForecastOutput
from visualization.charts import ChartBuilder


class VisualizationAgent:
    """Select and render charts for analysis outputs."""

    def __init__(self, settings: AppConfig) -> None:
        self.chart_builder = ChartBuilder(settings)

    def create_chart(self, dataframe: pd.DataFrame, question: str, dataset_name: str) -> dict[str, Any]:
        """Create a chart from a SQL result set."""

        return self.chart_builder.build_and_save(
            dataframe=dataframe,
            question=question,
            prefix=f"{dataset_name}_{question}",
            title=question,
        )

    def create_forecast_chart(
        self,
        forecast: ForecastOutput,
        question: str,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Create a chart for forecast output."""

        return self.chart_builder.build_and_save(
            dataframe=forecast.combined,
            question=question,
            prefix=f"{dataset_name}_forecast",
            kind="line",
            title=f"Forecast for {forecast.target_column}",
        )

    def create_cluster_chart(self, clusters: ClusterOutput, dataset_name: str) -> dict[str, Any]:
        """Create a cluster scatter chart."""

        frame = clusters.annotated_frame.copy()
        columns = clusters.feature_columns[:2]
        chart_frame = frame[columns + ["cluster"]]
        return self.chart_builder.build_and_save(
            dataframe=chart_frame,
            question="Cluster distribution",
            prefix=f"{dataset_name}_clusters",
            kind="scatter",
            title="Cluster Distribution",
        )

    def create_anomaly_chart(self, anomalies: AnomalyOutput, dataset_name: str) -> dict[str, Any]:
        """Create an anomaly visualization."""

        frame = anomalies.annotated_frame.copy()
        if len(anomalies.feature_columns) >= 2:
            chart_frame = frame[anomalies.feature_columns[:2] + ["anomaly_flag"]]
            return self.chart_builder.build_and_save(
                dataframe=chart_frame,
                question="Anomaly map",
                prefix=f"{dataset_name}_anomalies",
                kind="scatter",
                title="Anomaly Detection",
            )

        counts = (
            frame["anomaly_flag"]
            .map({1: "normal", -1: "anomaly"})
            .value_counts()
            .reset_index()
        )
        counts.columns = ["label", "count"]
        return self.chart_builder.build_and_save(
            dataframe=counts,
            question="Anomaly counts",
            prefix=f"{dataset_name}_anomaly_counts",
            kind="bar",
            title="Anomaly Counts",
        )

    def generate_profile_charts(self, dataframe: pd.DataFrame, dataset_name: str) -> list[dict[str, Any]]:
        """Generate automated EDA charts."""

        return self.chart_builder.build_profile_charts(dataframe, dataset_name)

    def export_report(
        self,
        title: str,
        analysis: str,
        insights: str,
        recommendations: str,
        data_preview: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Export a PDF report."""

        return self.chart_builder.export_report(
            title=title,
            analysis=analysis,
            insights=insights,
            recommendations=recommendations,
            data_preview=data_preview,
        )
