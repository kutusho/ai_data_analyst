"""Chart generation and artifact export."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from analytics.statistics import preferred_numeric_columns
from backend.config import AppConfig


class ChartBuilder:
    """Build and persist interactive charts."""

    def __init__(self, settings: AppConfig) -> None:
        self.settings = settings

    def _slugify(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
        return re.sub(r"_+", "_", cleaned).strip("_") or "chart"

    def _artifact_url(self, path: Path) -> str:
        relative_path = path.relative_to(self.settings.artifacts_dir)
        return f"/artifacts/{relative_path.as_posix()}"

    def _is_time_like(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        name = series.name.lower()
        return any(token in name for token in ("date", "time", "period", "year", "month"))

    def choose_chart_type(self, dataframe: pd.DataFrame, question: str) -> str:
        """Select a chart type from the data shape."""

        numeric_columns = list(dataframe.select_dtypes(include=np.number).columns)
        categorical_columns = [
            column
            for column in dataframe.columns
            if column not in numeric_columns
        ]
        lowered_question = question.lower()

        if "correlation" in lowered_question and len(numeric_columns) >= 2:
            return "heatmap"
        if len(dataframe.columns) == 1 and numeric_columns:
            return "histogram"
        if dataframe.shape[1] >= 2:
            first_column = dataframe.columns[0]
            if self._is_time_like(dataframe[first_column]):
                return "line"
        if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
            return "bar"
        if len(numeric_columns) >= 2:
            return "scatter"
        return "table"

    def build_figure(
        self,
        dataframe: pd.DataFrame,
        question: str,
        kind: str | None = None,
        title: str | None = None,
    ) -> tuple[go.Figure | None, str]:
        """Create a Plotly figure for the DataFrame."""

        if dataframe.empty:
            return None, "table"

        chart_type = kind or self.choose_chart_type(dataframe, question)
        title = title or question
        numeric_columns = list(dataframe.select_dtypes(include=np.number).columns)
        categorical_columns = [
            column
            for column in dataframe.columns
            if column not in numeric_columns
        ]

        if chart_type == "heatmap" and len(numeric_columns) >= 2:
            corr_df = dataframe[numeric_columns].corr(numeric_only=True)
            figure = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale="RdBu",
                title=title,
            )
            return figure, chart_type

        if chart_type == "histogram" and numeric_columns:
            figure = px.histogram(dataframe, x=numeric_columns[0], title=title)
            return figure, chart_type

        if chart_type == "line" and len(dataframe.columns) >= 2:
            color_column = None
            for column in dataframe.columns[2:]:
                if not pd.api.types.is_numeric_dtype(dataframe[column]):
                    color_column = column
                    break
            figure = px.line(
                dataframe,
                x=dataframe.columns[0],
                y=dataframe.columns[1],
                color=color_column,
                markers=True,
                title=title,
            )
            return figure, chart_type

        if chart_type == "bar" and categorical_columns and numeric_columns:
            figure = px.bar(
                dataframe,
                x=categorical_columns[0],
                y=numeric_columns[0],
                color=categorical_columns[1] if len(categorical_columns) > 1 else None,
                title=title,
            )
            return figure, chart_type

        if chart_type == "scatter" and len(numeric_columns) >= 2:
            figure = px.scatter(
                dataframe,
                x=numeric_columns[0],
                y=numeric_columns[1],
                color=dataframe.columns[2] if len(dataframe.columns) > 2 else None,
                title=title,
            )
            return figure, chart_type

        return None, "table"

    def save_figure(self, figure: go.Figure, prefix: str) -> dict[str, Any]:
        """Persist a chart as an HTML artifact."""

        chart_id = f"{self._slugify(prefix)}_{uuid.uuid4().hex[:10]}"
        output_path = self.settings.charts_dir / f"{chart_id}.html"
        figure.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=70, b=40))
        figure.write_html(str(output_path), include_plotlyjs="cdn")
        return {
            "chart_id": chart_id,
            "path": str(output_path),
            "url": self._artifact_url(output_path),
        }

    def build_and_save(
        self,
        dataframe: pd.DataFrame,
        question: str,
        prefix: str,
        kind: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Create and save a chart when possible."""

        figure, chart_type = self.build_figure(dataframe, question=question, kind=kind, title=title)
        if figure is None:
            return {"chart_type": "table", "chart_id": None, "url": None, "path": None}

        payload = self.save_figure(figure, prefix=prefix)
        payload["chart_type"] = chart_type
        return payload

    def build_profile_charts(self, dataframe: pd.DataFrame, dataset_name: str) -> list[dict[str, Any]]:
        """Generate profile charts for an uploaded dataset."""

        payloads: list[dict[str, Any]] = []
        numeric_columns = preferred_numeric_columns(dataframe)
        if len(numeric_columns) >= 2:
            corr_df = dataframe[numeric_columns].copy()
            payloads.append(
                self.build_and_save(
                    corr_df,
                    question=f"Correlation view for {dataset_name}",
                    prefix=f"{dataset_name}_correlation",
                    kind="heatmap",
                    title=f"Correlation Matrix: {dataset_name}",
                )
            )

        for column in numeric_columns[:2]:
            payloads.append(
                self.build_and_save(
                    dataframe[[column]].dropna(),
                    question=f"Distribution for {column}",
                    prefix=f"{dataset_name}_{column}_distribution",
                    kind="histogram",
                    title=f"Distribution of {column}",
                )
            )

        return [payload for payload in payloads if payload.get("url")]

    def export_report(
        self,
        title: str,
        analysis: str,
        insights: str,
        recommendations: str,
        data_preview: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Export a simple PDF report for download."""

        report_id = f"{self._slugify(title)}_{uuid.uuid4().hex[:10]}"
        output_path = self.settings.reports_dir / f"{report_id}.pdf"
        preview_text = "\n".join(str(row) for row in data_preview[:5]) or "No rows returned."

        with PdfPages(output_path) as pdf:
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.suptitle(title, fontsize=16, fontweight="bold")
            plt.figtext(0.07, 0.88, "Analysis", fontsize=12, fontweight="bold")
            plt.figtext(0.07, 0.84, analysis[:1800], fontsize=10, wrap=True)
            plt.figtext(0.07, 0.56, "Insights", fontsize=12, fontweight="bold")
            plt.figtext(0.07, 0.52, insights[:1200], fontsize=10, wrap=True)
            plt.figtext(0.07, 0.34, "Recommendations", fontsize=12, fontweight="bold")
            plt.figtext(0.07, 0.30, recommendations[:1200], fontsize=10, wrap=True)
            plt.figtext(0.07, 0.13, "Data Preview", fontsize=12, fontweight="bold")
            plt.figtext(0.07, 0.03, preview_text[:1800], fontsize=8, wrap=True)
            plt.axis("off")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

        return {
            "report_id": report_id,
            "path": str(output_path),
            "url": self._artifact_url(output_path),
        }
