"""Controllers and schemas for the API layer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile
from pydantic import BaseModel, Field

from agents.analysis_agent import DataAnalysisAgent
from agents.insight_agent import InsightAgent
from agents.orchestrator import OrchestratorAgent
from agents.sql_agent import SQLGenerationAgent
from agents.visualization_agent import VisualizationAgent
from backend.config import AppConfig
from database.connector import UniversalDataConnector
from database.query_executor import SafeQueryExecutor
from ml.training import MLService
from utils.logger import get_logger

logger = get_logger(__name__)


class QueryOptions(BaseModel):
    """Optional controls for analysis requests."""

    forecast_periods: int = Field(default=12, ge=1, le=60)


class QueryRequest(BaseModel):
    """Request model for natural language data queries."""

    dataset_name: str
    question: str = Field(min_length=3)
    options: QueryOptions = Field(default_factory=QueryOptions)


class QueryResponse(BaseModel):
    """Structured response returned by the query endpoint."""

    workflow: str
    analysis: str
    insights: str
    chart_url: str | None = None
    recommendations: str
    sql: str | None = None
    chart_type: str | None = None
    data_preview: list[dict[str, Any]] = Field(default_factory=list)
    analysis_details: dict[str, Any] = Field(default_factory=dict)
    explainability: list[str] = Field(default_factory=list)
    cached: bool = False
    report_url: str | None = None


class DatasetUploadResponse(BaseModel):
    """Response returned after dataset registration."""

    dataset_name: str
    table_name: str
    row_count: int
    columns: list[dict[str, str]]
    profile: dict[str, Any]
    charts: list[dict[str, Any]]
    message: str


class PlatformController:
    """Central application controller used by the API routes."""

    def __init__(self, settings: AppConfig) -> None:
        self.settings = settings
        self.connector = UniversalDataConnector(settings)
        self.query_executor = SafeQueryExecutor(settings, self.connector)
        self.analysis_agent = DataAnalysisAgent()
        self.visualization_agent = VisualizationAgent(settings)
        self.insight_agent = InsightAgent(settings)
        self.sql_agent = SQLGenerationAgent(settings)
        self.ml_service = MLService()
        self.orchestrator = OrchestratorAgent(
            connector=self.connector,
            query_executor=self.query_executor,
            sql_agent=self.sql_agent,
            analysis_agent=self.analysis_agent,
            visualization_agent=self.visualization_agent,
            insight_agent=self.insight_agent,
            ml_service=self.ml_service,
        )
        self._history = self._load_history()
        self.bootstrap_examples()

    def _load_history(self) -> dict[str, list[dict[str, Any]]]:
        if not self.settings.history_path.exists():
            return {"insights": [], "charts": []}
        return json.loads(self.settings.history_path.read_text(encoding="utf-8"))

    def _save_history(self) -> None:
        self.settings.history_path.write_text(
            json.dumps(self._history, indent=2),
            encoding="utf-8",
        )

    def _record_chart(self, dataset_name: str, question: str, chart_payload: dict[str, Any]) -> None:
        if not chart_payload.get("url"):
            return
        self._history["charts"].append(
            {
                "dataset_name": dataset_name,
                "question": question,
                "chart_type": chart_payload.get("chart_type"),
                "chart_url": chart_payload.get("url"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._history["charts"] = self._history["charts"][-50:]
        self._save_history()

    def _record_insight(self, dataset_name: str, question: str, response: QueryResponse) -> None:
        self._history["insights"].append(
            {
                "dataset_name": dataset_name,
                "question": question,
                "workflow": response.workflow,
                "analysis": response.analysis,
                "insights": response.insights,
                "recommendations": response.recommendations,
                "chart_url": response.chart_url,
                "report_url": response.report_url,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._history["insights"] = self._history["insights"][-50:]
        self._save_history()

    def bootstrap_examples(self) -> None:
        """Register bundled CSV datasets at startup."""

        for dataset_file in ("tourism_data.csv", "sales_data.csv"):
            path = self.settings.datasets_dir / dataset_file
            dataset_name = Path(dataset_file).stem
            try:
                self.connector.get_dataset(dataset_name)
            except KeyError:
                if path.exists():
                    self.connector.register_csv(path, dataset_name=dataset_name)

    async def handle_upload(
        self,
        file: UploadFile | None = None,
        dataset_name: str | None = None,
        connection_url: str | None = None,
        table_name: str | None = None,
    ) -> DatasetUploadResponse:
        """Register a dataset from a CSV file or external database table."""

        if file is not None:
            target_path = self.settings.upload_dir / file.filename
            target_path.write_bytes(await file.read())
            metadata = self.connector.register_csv(
                target_path,
                dataset_name=dataset_name or Path(file.filename).stem,
            )
        elif connection_url and table_name and dataset_name:
            metadata = self.connector.register_external_table(
                dataset_name=dataset_name,
                connection_url=connection_url,
                table_name=table_name,
            )
        else:
            raise ValueError("Provide either a CSV file or a connection URL plus table name.")

        dataframe = self.query_executor.fetch_dataset_frame(metadata.name)
        profile = self.analysis_agent.profile_dataset(dataframe)
        charts = self.visualization_agent.generate_profile_charts(dataframe, metadata.name)
        for chart in charts:
            self._record_chart(metadata.name, "Automated dataset profile", chart)

        return DatasetUploadResponse(
            dataset_name=metadata.name,
            table_name=metadata.table_name,
            row_count=metadata.row_count,
            columns=metadata.columns,
            profile=profile,
            charts=charts,
            message=f"Dataset '{metadata.name}' registered successfully.",
        )

    def handle_query(self, payload: QueryRequest) -> QueryResponse:
        """Run the end-to-end analyst workflow and export artifacts."""

        orchestration_result = self.orchestrator.run(
            dataset_name=payload.dataset_name,
            question=payload.question,
            options=payload.options.model_dump(),
        )
        report = self.visualization_agent.export_report(
            title=payload.question,
            analysis=orchestration_result["analysis"],
            insights=orchestration_result["insights"],
            recommendations=orchestration_result["recommendations"],
            data_preview=orchestration_result["data_preview"],
        )
        orchestration_result["report_url"] = report["url"]
        response = QueryResponse.model_validate(orchestration_result)

        if response.chart_url:
            self._record_chart(
                dataset_name=payload.dataset_name,
                question=payload.question,
                chart_payload={
                    "url": response.chart_url,
                    "chart_type": response.chart_type,
                },
            )
        self._record_insight(payload.dataset_name, payload.question, response)
        return response

    def get_insights(self, limit: int = 20) -> dict[str, Any]:
        """Return recent insight artifacts."""

        return {"items": list(reversed(self._history["insights"][-limit:]))}

    def get_charts(self, limit: int = 20) -> dict[str, Any]:
        """Return recent chart artifacts."""

        return {"items": list(reversed(self._history["charts"][-limit:]))}

    def list_datasets(self) -> dict[str, Any]:
        """Return the registered datasets."""

        return {
            "items": [
                metadata.model_dump(mode="json")
                for metadata in self.connector.list_datasets()
            ]
        }
