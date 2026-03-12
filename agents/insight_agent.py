"""Agent responsible for writing insights and recommendations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.config import AppConfig
from utils.llm import OpenAIClient
from utils.prompts import build_insight_prompt


class InsightAgent:
    """Generate grounded analytical insights."""

    def __init__(self, settings: AppConfig) -> None:
        self.llm_client = OpenAIClient(settings)

    def generate_insights(
        self,
        question: str,
        analysis_summary: str,
        evidence: list[str],
        dataframe: pd.DataFrame,
        workflow: str,
    ) -> dict[str, Any]:
        """Return insight text, recommendations, and explainability notes."""

        sample_rows = dataframe.head(5).replace({pd.NA: None}).to_dict(orient="records")
        insight_text = analysis_summary

        if self.llm_client.available:
            prompt = build_insight_prompt(
                question=question,
                analysis_summary=analysis_summary,
                evidence=evidence,
                sample_rows=sample_rows,
            )
            instructions = (
                "You are a data analyst. Write one short paragraph of evidence-based insight. "
                "Do not invent facts and do not add bullet points."
            )
            result = self.llm_client.generate_text(instructions=instructions, user_input=prompt)
            if result.text:
                insight_text = result.text

        recommendations = self._recommendations_from_workflow(workflow, evidence)
        return {
            "insights": insight_text,
            "recommendations": recommendations,
            "explainability": evidence[:5],
        }

    def _recommendations_from_workflow(self, workflow: str, evidence: list[str]) -> str:
        if workflow == "forecast":
            return (
                "Use the forecast as a capacity-planning baseline, validate it against upcoming events, "
                "and monitor deviation from the projected trend each month."
            )
        if workflow == "cluster":
            return (
                "Turn the identified segments into audience tiers, design differentiated campaigns for each cluster, "
                "and validate whether the clusters map to real commercial behavior."
            )
        if workflow == "anomaly":
            return (
                "Review the flagged records for operational issues or data quality problems, then add alert thresholds "
                "for the numeric drivers used in the anomaly model."
            )
        if evidence:
            return (
                "Double down on the leading segment shown in the results, investigate the drivers behind weaker segments, "
                "and track the highlighted metric in a recurring dashboard."
            )
        return "Collect more data or refine the question to produce a sharper recommendation."
