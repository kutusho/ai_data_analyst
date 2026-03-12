"""Agent responsible for dataset analysis and EDA synthesis."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from analytics.eda import EDAService
from analytics.statistics import preferred_numeric_columns
from ml.models import AnomalyOutput, ClusterOutput, ForecastOutput


class DataAnalysisAgent:
    """Analyze result sets and summarize statistical findings."""

    def __init__(self) -> None:
        self.eda_service = EDAService()

    def infer_metric_column(self, question: str, schema: list[dict[str, Any]]) -> str | None:
        """Infer the most likely target metric from a question."""

        lowered_question = question.lower()
        question_tokens = set(re.findall(r"[a-zA-Z_]+", lowered_question))
        synonym_groups = {
            "revenue": {"revenue", "sales", "income", "turnover"},
            "visitors": {"visitors", "visitor", "tourists", "tourism", "demand"},
            "profit": {"profit", "margin"},
            "quantity": {"quantity", "units", "volume"},
        }
        available_columns = [column["name"] for column in schema]
        numeric_columns = [
            column["name"]
            for column in schema
            if any(token in column["dtype"].lower() for token in ("int", "float", "double", "numeric"))
        ]
        preferred_numeric = [
            column
            for column in numeric_columns
            if column.lower() not in {"year", "month"} and not column.lower().endswith("_id")
        ]

        for column in preferred_numeric:
            if column.lower() in lowered_question:
                return column

        for preferred_column, keywords in synonym_groups.items():
            if keywords & question_tokens:
                for column in preferred_numeric:
                    if preferred_column in column.lower():
                        return column

        return preferred_numeric[0] if preferred_numeric else (numeric_columns[0] if numeric_columns else None)

    def profile_dataset(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        """Generate an EDA profile for a full dataset."""

        return self.eda_service.run(dataframe)

    def analyze_query_result(self, dataframe: pd.DataFrame, question: str) -> dict[str, Any]:
        """Analyze a query result and create explainable evidence."""

        details = self.eda_service.run(dataframe)
        evidence: list[str] = []
        summary_lines = [
            f"The query returned {len(dataframe):,} rows and {len(dataframe.columns)} columns."
        ]

        if dataframe.empty:
            summary_lines.append("No data matched the query filters.")
            return {"summary_text": " ".join(summary_lines), "evidence": evidence, "details": details}

        numeric_columns = preferred_numeric_columns(dataframe)
        all_numeric_columns = list(dataframe.select_dtypes(include=np.number).columns)
        categorical_columns = [
            column
            for column in dataframe.columns
            if column not in all_numeric_columns
        ]

        if numeric_columns:
            primary_metric = numeric_columns[0]
            total_value = float(pd.to_numeric(dataframe[primary_metric], errors="coerce").fillna(0).sum())
            avg_value = float(pd.to_numeric(dataframe[primary_metric], errors="coerce").fillna(0).mean())
            summary_lines.append(
                f"The primary metric `{primary_metric}` totals {total_value:,.2f} with an average of {avg_value:,.2f}."
            )
            evidence.append(f"{primary_metric} total: {total_value:,.2f}")
            evidence.append(f"{primary_metric} average: {avg_value:,.2f}")

            if categorical_columns:
                ranked = dataframe.sort_values(primary_metric, ascending=False).head(1)
                top_row = ranked.iloc[0].to_dict()
                dimension = categorical_columns[0]
                summary_lines.append(
                    f"The top `{dimension}` is `{top_row.get(dimension)}` with {top_row.get(primary_metric):,.2f}."
                )
                evidence.append(
                    f"Top {dimension}: {top_row.get(dimension)} ({primary_metric}={top_row.get(primary_metric):,.2f})"
                )

        trend = details.get("trend_analysis", {})
        if trend:
            percent_change = trend.get("percent_change")
            if percent_change is not None:
                summary_lines.append(
                    f"The time trend changed by {percent_change:,.2f}% between {trend['first_period']} and {trend['last_period']}."
                )
                evidence.append(
                    f"Trend change: {percent_change:,.2f}% from {trend['first_period']} to {trend['last_period']}"
                )

        return {
            "summary_text": " ".join(summary_lines),
            "evidence": evidence,
            "details": details,
        }

    def summarize_forecast(self, forecast: ForecastOutput) -> dict[str, Any]:
        """Create a narrative summary for a forecast."""

        history_last = float(forecast.history[forecast.target_column].iloc[-1])
        forecast_last = float(forecast.forecast[forecast.target_column].iloc[-1])
        change = forecast_last - history_last
        pct_change = (change / history_last * 100) if history_last else 0.0
        evidence = [
            f"Forecast target: {forecast.target_column}",
            f"Last historical value: {history_last:,.2f}",
            f"Last forecast value: {forecast_last:,.2f}",
            f"Model R2 score: {forecast.r2_score:,.4f}",
        ]
        summary = (
            f"The forecast projects `{forecast.target_column}` from {history_last:,.2f} "
            f"to {forecast_last:,.2f} over the next {len(forecast.forecast)} periods, "
            f"a change of {pct_change:,.2f}%."
        )
        return {
            "summary_text": summary,
            "evidence": evidence,
            "details": {
                "model_type": forecast.model_type,
                "r2_score": forecast.r2_score,
                "forecast_rows": forecast.forecast.to_dict(orient="records"),
            },
        }

    def summarize_clusters(self, clusters: ClusterOutput) -> dict[str, Any]:
        """Create a narrative summary for clustering output."""

        cluster_sizes = clusters.annotated_frame["cluster"].value_counts().sort_index().to_dict()
        summary = (
            f"The clustering workflow created {clusters.cluster_count} segments using "
            f"{', '.join(clusters.feature_columns)}."
        )
        evidence = [
            f"Cluster {cluster_id}: {size} rows"
            for cluster_id, size in cluster_sizes.items()
        ]
        return {
            "summary_text": summary,
            "evidence": evidence,
            "details": {"cluster_sizes": cluster_sizes, "feature_columns": clusters.feature_columns},
        }

    def summarize_anomalies(self, anomalies: AnomalyOutput) -> dict[str, Any]:
        """Create a narrative summary for anomaly detection output."""

        summary = (
            f"The anomaly detection workflow flagged {anomalies.anomaly_count} anomalous rows "
            f"using {', '.join(anomalies.feature_columns)}."
        )
        evidence = [
            f"Anomalies detected: {anomalies.anomaly_count}",
            f"Feature columns: {', '.join(anomalies.feature_columns)}",
        ]
        sample = anomalies.annotated_frame[anomalies.annotated_frame["anomaly_flag"] == -1].head(10)
        return {
            "summary_text": summary,
            "evidence": evidence,
            "details": {"sample_anomalies": sample.to_dict(orient="records")},
        }
