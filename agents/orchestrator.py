"""Orchestration logic for the AI Data Analyst Platform."""

from __future__ import annotations

from typing import Any

from agents.analysis_agent import DataAnalysisAgent
from agents.insight_agent import InsightAgent
from agents.sql_agent import SQLGenerationAgent
from agents.visualization_agent import VisualizationAgent
from database.connector import UniversalDataConnector
from database.query_executor import SafeQueryExecutor
from ml.training import MLService


class OrchestratorAgent:
    """Coordinate the specialized agents for a user request."""

    def __init__(
        self,
        connector: UniversalDataConnector,
        query_executor: SafeQueryExecutor,
        sql_agent: SQLGenerationAgent,
        analysis_agent: DataAnalysisAgent,
        visualization_agent: VisualizationAgent,
        insight_agent: InsightAgent,
        ml_service: MLService,
    ) -> None:
        self.connector = connector
        self.query_executor = query_executor
        self.sql_agent = sql_agent
        self.analysis_agent = analysis_agent
        self.visualization_agent = visualization_agent
        self.insight_agent = insight_agent
        self.ml_service = ml_service

    def run(
        self,
        dataset_name: str,
        question: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the appropriate workflow for the question."""

        resolved_options = options or {}
        workflow = self._detect_workflow(question)
        if workflow == "forecast":
            return self._run_forecast(dataset_name, question, resolved_options)
        if workflow == "cluster":
            return self._run_cluster(dataset_name, question)
        if workflow == "anomaly":
            return self._run_anomaly(dataset_name, question)
        return self._run_sql_analysis(dataset_name, question)

    def _detect_workflow(self, question: str) -> str:
        lowered_question = question.lower()
        if any(token in lowered_question for token in ("predict", "forecast", "projection", "next year")):
            return "forecast"
        if any(token in lowered_question for token in ("cluster", "segment", "segmentation")):
            return "cluster"
        if any(token in lowered_question for token in ("anomaly", "anomalies", "outlier")):
            return "anomaly"
        return "sql_analysis"

    def _run_sql_analysis(self, dataset_name: str, question: str) -> dict[str, Any]:
        dataset = self.connector.get_dataset(dataset_name)
        sql = self.sql_agent.generate_sql(question, dataset)
        result_df, cached = self.query_executor.execute(dataset_name, sql, use_cache=True)
        analysis = self.analysis_agent.analyze_query_result(result_df, question)
        chart = self.visualization_agent.create_chart(result_df, question, dataset_name)
        insight_payload = self.insight_agent.generate_insights(
            question=question,
            analysis_summary=analysis["summary_text"],
            evidence=analysis["evidence"],
            dataframe=result_df,
            workflow="analysis",
        )
        return {
            "workflow": "sql_analysis",
            "sql": sql,
            "analysis": analysis["summary_text"],
            "analysis_details": analysis["details"],
            "insights": insight_payload["insights"],
            "recommendations": insight_payload["recommendations"],
            "chart_url": chart.get("url"),
            "chart_type": chart.get("chart_type"),
            "data_preview": self.query_executor.to_serializable_records(result_df),
            "explainability": insight_payload["explainability"],
            "cached": cached,
        }

    def _run_forecast(
        self,
        dataset_name: str,
        question: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        dataset = self.connector.get_dataset(dataset_name)
        dataframe = self.query_executor.fetch_dataset_frame(dataset_name)
        target_column = self.analysis_agent.infer_metric_column(question, dataset.columns)
        if not target_column:
            raise ValueError("The platform could not infer a target metric for forecasting.")

        forecast = self.ml_service.time_series_projection(
            dataframe=dataframe,
            target_column=target_column,
            periods=int(options.get("forecast_periods", 12)),
        )
        analysis = self.analysis_agent.summarize_forecast(forecast)
        chart = self.visualization_agent.create_forecast_chart(forecast, question, dataset_name)
        insight_payload = self.insight_agent.generate_insights(
            question=question,
            analysis_summary=analysis["summary_text"],
            evidence=analysis["evidence"],
            dataframe=forecast.combined,
            workflow="forecast",
        )
        return {
            "workflow": "forecast",
            "sql": None,
            "analysis": analysis["summary_text"],
            "analysis_details": analysis["details"],
            "insights": insight_payload["insights"],
            "recommendations": insight_payload["recommendations"],
            "chart_url": chart.get("url"),
            "chart_type": chart.get("chart_type"),
            "data_preview": forecast.forecast.head(12).to_dict(orient="records"),
            "explainability": insight_payload["explainability"],
            "cached": False,
        }

    def _run_cluster(self, dataset_name: str, question: str) -> dict[str, Any]:
        dataframe = self.query_executor.fetch_dataset_frame(dataset_name)
        clusters = self.ml_service.cluster_dataset(dataframe)
        analysis = self.analysis_agent.summarize_clusters(clusters)
        chart = self.visualization_agent.create_cluster_chart(clusters, dataset_name)
        insight_payload = self.insight_agent.generate_insights(
            question=question,
            analysis_summary=analysis["summary_text"],
            evidence=analysis["evidence"],
            dataframe=clusters.annotated_frame,
            workflow="cluster",
        )
        return {
            "workflow": "cluster",
            "sql": None,
            "analysis": analysis["summary_text"],
            "analysis_details": analysis["details"],
            "insights": insight_payload["insights"],
            "recommendations": insight_payload["recommendations"],
            "chart_url": chart.get("url"),
            "chart_type": chart.get("chart_type"),
            "data_preview": clusters.annotated_frame.head(20).to_dict(orient="records"),
            "explainability": insight_payload["explainability"],
            "cached": False,
        }

    def _run_anomaly(self, dataset_name: str, question: str) -> dict[str, Any]:
        dataframe = self.query_executor.fetch_dataset_frame(dataset_name)
        anomalies = self.ml_service.detect_anomalies(dataframe)
        analysis = self.analysis_agent.summarize_anomalies(anomalies)
        chart = self.visualization_agent.create_anomaly_chart(anomalies, dataset_name)
        insight_payload = self.insight_agent.generate_insights(
            question=question,
            analysis_summary=analysis["summary_text"],
            evidence=analysis["evidence"],
            dataframe=anomalies.annotated_frame,
            workflow="anomaly",
        )
        return {
            "workflow": "anomaly",
            "sql": None,
            "analysis": analysis["summary_text"],
            "analysis_details": analysis["details"],
            "insights": insight_payload["insights"],
            "recommendations": insight_payload["recommendations"],
            "chart_url": chart.get("url"),
            "chart_type": chart.get("chart_type"),
            "data_preview": anomalies.annotated_frame.head(20).to_dict(orient="records"),
            "explainability": insight_payload["explainability"],
            "cached": False,
        }
