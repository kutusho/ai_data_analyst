"""Safe query execution and result caching."""

from __future__ import annotations

import hashlib
import io
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import text

from backend.config import AppConfig
from database.connector import UniversalDataConnector

FORBIDDEN_SQL_TOKENS = {
    "alter",
    "attach",
    "copy",
    "create",
    "delete",
    "detach",
    "drop",
    "grant",
    "insert",
    "merge",
    "replace",
    "revoke",
    "truncate",
    "update",
    "vacuum",
}


class SafeQueryExecutor:
    """Execute read-only SQL with SQLite-backed caching."""

    def __init__(self, settings: AppConfig, connector: UniversalDataConnector) -> None:
        self.settings = settings
        self.connector = connector
        self._ensure_cache_table()

    def _ensure_cache_table(self) -> None:
        with sqlite3.connect(self.settings.sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    sql_text TEXT NOT NULL,
                    dataframe_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def validate_read_only_sql(self, sql: str) -> str:
        """Validate that the provided SQL is read-only."""

        normalized = sql.strip().strip(";")
        lowered = normalized.lower()
        if not lowered.startswith(("select", "with")):
            raise ValueError("Only SELECT and CTE queries are allowed.")
        if ";" in normalized:
            raise ValueError("Only single-statement queries are allowed.")
        forbidden_pattern = re.compile(r"\b(" + "|".join(sorted(FORBIDDEN_SQL_TOKENS)) + r")\b")
        if forbidden_pattern.search(lowered):
            raise ValueError("Potentially destructive SQL was rejected.")
        return normalized

    def _cache_key(self, dataset_name: str, sql: str) -> str:
        digest = hashlib.sha256(f"{dataset_name}:{sql}".encode("utf-8")).hexdigest()
        return digest

    def get_cached_result(self, dataset_name: str, sql: str) -> pd.DataFrame | None:
        """Return a cached DataFrame if present."""

        cache_key = self._cache_key(dataset_name, sql)
        with sqlite3.connect(self.settings.sqlite_path) as connection:
            row = connection.execute(
                "SELECT dataframe_json FROM query_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        if not row:
            return None
        return pd.read_json(io.StringIO(row[0]), orient="split")

    def set_cached_result(self, dataset_name: str, sql: str, dataframe: pd.DataFrame) -> None:
        """Persist a query result in the local cache."""

        cache_key = self._cache_key(dataset_name, sql)
        with sqlite3.connect(self.settings.sqlite_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO query_cache (
                    cache_key,
                    dataset_name,
                    sql_text,
                    dataframe_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    dataset_name,
                    sql,
                    dataframe.to_json(orient="split", date_format="iso"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()

    def execute(
        self,
        dataset_name: str,
        sql: str,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame, bool]:
        """Execute SQL for the given dataset."""

        safe_sql = self.validate_read_only_sql(sql)
        if use_cache:
            cached = self.get_cached_result(dataset_name, safe_sql)
            if cached is not None:
                return cached, True

        engine = self.connector.connect(dataset_name)
        with engine.connect() as connection:
            dataframe = pd.read_sql_query(text(safe_sql), connection)

        self.set_cached_result(dataset_name, safe_sql, dataframe)
        return dataframe, False

    def fetch_dataset_frame(
        self,
        dataset_name: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load rows directly from a registered dataset table."""

        metadata = self.connector.get_dataset(dataset_name)
        sql = f'SELECT * FROM "{metadata.table_name}"'
        if limit:
            sql = f"{sql} LIMIT {limit}"
        dataframe, _ = self.execute(dataset_name, sql, use_cache=False)
        return dataframe

    def to_serializable_records(self, dataframe: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
        """Convert a DataFrame into JSON-safe records for API responses."""

        return dataframe.head(limit).replace({pd.NA: None}).to_dict(orient="records")
