"""Agent responsible for generating safe SQL from natural language."""

from __future__ import annotations

import re
from typing import Any

from backend.config import AppConfig
from database.connector import DatasetMetadata, sanitize_identifier
from utils.llm import OpenAIClient
from utils.prompts import build_sql_generation_prompt


class SQLGenerationAgent:
    """Generate and validate SQL queries from user questions."""

    def __init__(self, settings: AppConfig) -> None:
        self.settings = settings
        self.llm_client = OpenAIClient(settings)

    def generate_sql(self, question: str, dataset: DatasetMetadata) -> str:
        """Return a safe read-only SQL query."""

        if self.llm_client.available:
            instructions = (
                "You are a senior analytics engineer. Return only SQL with no markdown, "
                "no prose, and no destructive statements."
            )
            prompt = build_sql_generation_prompt(
                question=question,
                table_name=dataset.table_name,
                schema=dataset.columns,
            )
            llm_result = self.llm_client.generate_text(instructions=instructions, user_input=prompt)
            if llm_result.text:
                cleaned = self._clean_sql(llm_result.text)
                if self.is_safe_sql(cleaned):
                    return cleaned

        return self._heuristic_sql(question, dataset)

    def _clean_sql(self, sql: str) -> str:
        cleaned = sql.strip()
        cleaned = cleaned.removeprefix("```sql").removeprefix("```").removesuffix("```")
        return cleaned.strip().strip(";")

    def is_safe_sql(self, sql: str) -> bool:
        lowered = sql.lower().strip()
        if not lowered.startswith(("select", "with")):
            return False
        if ";" in lowered:
            return False
        forbidden_pattern = re.compile(
            r"\b(alter|attach|copy|create|delete|detach|drop|grant|insert|merge|replace|revoke|truncate|update|vacuum)\b"
        )
        return forbidden_pattern.search(lowered) is None

    def _quote(self, identifier: str) -> str:
        return f'"{identifier}"'

    def _heuristic_sql(self, question: str, dataset: DatasetMetadata) -> str:
        lowered_question = question.lower()
        columns = [column["name"] for column in dataset.columns]
        quoted_table = self._quote(dataset.table_name)
        numeric_columns = [
            column["name"]
            for column in dataset.columns
            if any(token in column["dtype"].lower() for token in ("int", "float", "double", "numeric"))
        ]
        preferred_numeric = [
            column
            for column in numeric_columns
            if column.lower() not in {"year", "month"} and not column.lower().endswith("_id")
        ]
        metric = self._infer_metric(lowered_question, columns, numeric_columns)
        group_columns = self._infer_group_columns(lowered_question, columns)
        where_clauses = self._infer_filters(lowered_question, columns)
        limit = self._infer_limit(lowered_question)

        if "correlation" in lowered_question and len(preferred_numeric) >= 2:
            selected = ", ".join(self._quote(column) for column in preferred_numeric[:5])
            return f"SELECT {selected} FROM {quoted_table} LIMIT 1000"

        if "missing" in lowered_question:
            missing_columns = ", ".join(
                f"SUM(CASE WHEN {self._quote(column)} IS NULL THEN 1 ELSE 0 END) AS {sanitize_identifier(column)}_missing"
                for column in columns
            )
            return f"SELECT {missing_columns} FROM {quoted_table}"

        if "distribution" in lowered_question and metric:
            return (
                f"SELECT {self._quote(metric)} FROM {quoted_table} "
                f"WHERE {self._quote(metric)} IS NOT NULL LIMIT 2000"
            )

        aggregate = self._infer_aggregate(lowered_question, metric, bool(group_columns))
        filters_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        if group_columns:
            if metric:
                alias = sanitize_identifier(f"{aggregate}_{metric}")
                if aggregate == "count":
                    metric_expr = "COUNT(*)"
                    alias = "record_count"
                else:
                    metric_expr = f"{aggregate.upper()}({self._quote(metric)})"
                select_items = ", ".join(
                    [*(self._quote(column) for column in group_columns), f"{metric_expr} AS {alias}"]
                )
                sql = (
                    f"SELECT {select_items} FROM {quoted_table}{filters_sql} "
                    f"GROUP BY {', '.join(self._quote(column) for column in group_columns)}"
                )
                if any(token in lowered_question for token in ("highest", "most", "top", "largest")):
                    sql += f" ORDER BY {alias} DESC LIMIT {limit}"
                elif any(token in lowered_question for token in ("lowest", "smallest", "least")):
                    sql += f" ORDER BY {alias} ASC LIMIT {limit}"
                elif any(token in lowered_question for token in ("trend", "growth", "monthly", "yearly")):
                    sql += f" ORDER BY {', '.join(self._quote(column) for column in group_columns)}"
                else:
                    sql += f" ORDER BY {alias} DESC"
                return sql

            select_items = ", ".join([*(self._quote(column) for column in group_columns), "COUNT(*) AS record_count"])
            return (
                f"SELECT {select_items} FROM {quoted_table}{filters_sql} "
                f"GROUP BY {', '.join(self._quote(column) for column in group_columns)} "
                "ORDER BY record_count DESC"
            )

        if metric:
            if any(token in lowered_question for token in ("average", "avg", "mean")):
                return f"SELECT AVG({self._quote(metric)}) AS average_{sanitize_identifier(metric)} FROM {quoted_table}{filters_sql}"
            if any(token in lowered_question for token in ("highest", "maximum", "max")):
                return f"SELECT MAX({self._quote(metric)}) AS max_{sanitize_identifier(metric)} FROM {quoted_table}{filters_sql}"
            if any(token in lowered_question for token in ("lowest", "minimum", "min")):
                return f"SELECT MIN({self._quote(metric)}) AS min_{sanitize_identifier(metric)} FROM {quoted_table}{filters_sql}"
            if any(token in lowered_question for token in ("total", "sum")):
                return f"SELECT SUM({self._quote(metric)}) AS total_{sanitize_identifier(metric)} FROM {quoted_table}{filters_sql}"
            return (
                f"SELECT {', '.join(self._quote(column) for column in columns[: min(5, len(columns))])} "
                f"FROM {quoted_table}{filters_sql} LIMIT {limit}"
            )

        return f"SELECT * FROM {quoted_table}{filters_sql} LIMIT {limit}"

    def _infer_metric(
        self,
        question: str,
        columns: list[str],
        numeric_columns: list[str],
    ) -> str | None:
        synonyms = {
            "revenue": {"revenue", "sales", "income", "turnover"},
            "visitors": {"visitors", "visitor", "tourists", "tourism", "demand"},
            "profit": {"profit", "margin"},
            "quantity": {"quantity", "units", "volume"},
        }
        question_tokens = set(re.findall(r"[a-zA-Z_]+", question))
        preferred_numeric = [
            column
            for column in numeric_columns
            if column.lower() not in {"year", "month"} and not column.lower().endswith("_id")
        ]

        for column in preferred_numeric:
            if column.lower() in question:
                return column

        for preferred, keywords in synonyms.items():
            if keywords & question_tokens:
                for column in preferred_numeric:
                    if preferred in column.lower():
                        return column
        return preferred_numeric[0] if preferred_numeric else (numeric_columns[0] if numeric_columns else None)

    def _infer_group_columns(self, question: str, columns: list[str]) -> list[str]:
        group_columns: list[str] = []
        lowered_columns = {column.lower(): column for column in columns}

        if "month" in question and "month" in lowered_columns:
            if "year" in lowered_columns:
                group_columns.append(lowered_columns["year"])
            group_columns.append(lowered_columns["month"])
        elif "year" in question and "year" in lowered_columns:
            group_columns.append(lowered_columns["year"])

        for keyword, fallback in (
            ("region", "region"),
            ("destination", "region"),
            ("category", "category"),
            ("product", "product"),
            ("segment", "segment"),
        ):
            if keyword in question and fallback in lowered_columns and lowered_columns[fallback] not in group_columns:
                group_columns.append(lowered_columns[fallback])

        by_match = re.search(r"\bby\s+([a-zA-Z_]+)", question)
        if by_match:
            candidate = by_match.group(1)
            if candidate in lowered_columns and lowered_columns[candidate] not in group_columns:
                group_columns.append(lowered_columns[candidate])

        if not group_columns and any(token in question for token in ("highest", "most", "top")) and "region" in lowered_columns:
            group_columns.append(lowered_columns["region"])

        return group_columns

    def _infer_filters(self, question: str, columns: list[str]) -> list[str]:
        where_clauses: list[str] = []
        lowered_columns = {column.lower(): column for column in columns}

        if "year" in lowered_columns:
            years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
            if years:
                joined = ", ".join(years)
                where_clauses.append(f'{self._quote(lowered_columns["year"])} IN ({joined})')

        if "month" in lowered_columns:
            month_match = re.search(
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                question,
            )
            if month_match:
                month_name = month_match.group(1).capitalize()
                where_clauses.append(f"{self._quote(lowered_columns['month'])} = '{month_name}'")

        return where_clauses

    def _infer_aggregate(self, question: str, metric: str | None, has_grouping: bool) -> str:
        if any(token in question for token in ("average", "avg", "mean")):
            return "avg"
        if any(token in question for token in ("count", "how many", "number of")):
            return "count"
        if any(token in question for token in ("maximum", "max")) and not has_grouping:
            return "max"
        if any(token in question for token in ("minimum", "min")) and not has_grouping:
            return "min"
        if metric:
            return "sum"
        return "count"

    def _infer_limit(self, question: str) -> int:
        match = re.search(r"\btop\s+(\d+)\b", question)
        if match:
            return int(match.group(1))
        return 10 if any(token in question for token in ("top", "highest", "lowest", "most")) else 200
