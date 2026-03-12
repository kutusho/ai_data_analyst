"""Prompt builders for the LLM-backed agents."""

from __future__ import annotations

from typing import Any


def format_schema(schema: list[dict[str, Any]]) -> str:
    """Convert a table schema into a compact prompt string."""

    return "\n".join(
        f"- {column['name']} ({column['dtype']})"
        for column in schema
    )


def build_sql_generation_prompt(
    question: str,
    table_name: str,
    schema: list[dict[str, Any]],
) -> str:
    """Create a SQL generation prompt from the user question and schema."""

    return (
        f"User question: {question}\n"
        f"Target table: {table_name}\n"
        "Table schema:\n"
        f"{format_schema(schema)}\n\n"
        "Requirements:\n"
        "- Use ANSI SQL compatible with PostgreSQL or SQLite.\n"
        "- Return a single read-only query.\n"
        "- Never use INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE.\n"
        "- Prefer meaningful aliases.\n"
        "- If the request implies aggregation, group explicitly.\n"
        "- If the question asks for ranking, sort by the key metric.\n"
        "- If the question is broad, keep the result compact and useful."
    )


def build_insight_prompt(
    question: str,
    analysis_summary: str,
    evidence: list[str],
    sample_rows: list[dict[str, Any]],
) -> str:
    """Create an insight synthesis prompt."""

    return (
        f"User question: {question}\n\n"
        f"Analysis summary:\n{analysis_summary}\n\n"
        "Evidence points:\n"
        f"{chr(10).join(f'- {item}' for item in evidence) or '- No evidence provided.'}\n\n"
        "Sample rows:\n"
        f"{sample_rows}\n\n"
        "Write a concise business insight paragraph followed by 2 actionable "
        "recommendations. Ground every statement in the evidence and avoid speculation."
    )
