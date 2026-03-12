"""Universal data connector for CSV, SQLite, and PostgreSQL datasets."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

from backend.config import AppConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def sanitize_identifier(value: str) -> str:
    """Return a SQL-safe identifier."""

    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "dataset"


class DatasetMetadata(BaseModel):
    """Metadata describing a registered dataset."""

    name: str
    table_name: str
    source_type: str
    source_path: str | None = None
    database_url: str
    row_count: int = 0
    columns: list[dict[str, str]] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list)
    uploaded_at: str


class UniversalDataConnector:
    """Manage dataset registration and SQLAlchemy connections."""

    def __init__(self, settings: AppConfig) -> None:
        self.settings = settings
        self._engine_cache: dict[str, Engine] = {}
        self._registry = self._load_registry()

    def _load_registry(self) -> dict[str, DatasetMetadata]:
        if not self.settings.dataset_registry_path.exists():
            return {}

        raw = json.loads(self.settings.dataset_registry_path.read_text(encoding="utf-8"))
        return {
            name: DatasetMetadata.model_validate(metadata)
            for name, metadata in raw.items()
        }

    def _save_registry(self) -> None:
        payload = {
            name: metadata.model_dump(mode="json")
            for name, metadata in self._registry.items()
        }
        self.settings.dataset_registry_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _get_engine(self, database_url: str) -> Engine:
        if database_url not in self._engine_cache:
            self._engine_cache[database_url] = create_engine(database_url, future=True)
        return self._engine_cache[database_url]

    def list_datasets(self) -> list[DatasetMetadata]:
        """Return all registered datasets."""

        return sorted(self._registry.values(), key=lambda item: item.name)

    def get_dataset(self, dataset_name: str) -> DatasetMetadata:
        """Return metadata for the requested dataset."""

        dataset_key = sanitize_identifier(dataset_name)
        metadata = self._registry.get(dataset_key)
        if metadata is None:
            raise KeyError(f"Dataset '{dataset_name}' is not registered.")
        return metadata

    def connect(self, dataset_name: str) -> Engine:
        """Return a SQLAlchemy engine for a registered dataset."""

        metadata = self.get_dataset(dataset_name)
        return self._get_engine(metadata.database_url)

    def get_schema(self, dataset_name: str) -> list[dict[str, str]]:
        """Return the cached schema for a dataset."""

        return self.get_dataset(dataset_name).columns

    def register_csv(self, file_path: Path, dataset_name: str | None = None) -> DatasetMetadata:
        """Register a CSV file by loading it into the managed SQLite warehouse."""

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        resolved_name = sanitize_identifier(dataset_name or file_path.stem)
        table_name = sanitize_identifier(resolved_name)
        dataframe = pd.read_csv(file_path)
        sqlite_engine = self._get_engine(self.settings.default_database_url)
        dataframe.to_sql(table_name, sqlite_engine, if_exists="replace", index=False)

        metadata = DatasetMetadata(
            name=resolved_name,
            table_name=table_name,
            source_type="csv",
            source_path=str(file_path),
            database_url=self.settings.default_database_url,
            row_count=len(dataframe),
            columns=[
                {"name": column, "dtype": str(dtype)}
                for column, dtype in dataframe.dtypes.items()
            ],
            preview=dataframe.head(5).replace({pd.NA: None}).to_dict(orient="records"),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        self._registry[resolved_name] = metadata
        self._save_registry()
        logger.info("Registered CSV dataset '%s' -> table '%s'", resolved_name, table_name)
        return metadata

    def register_external_table(
        self,
        dataset_name: str,
        connection_url: str,
        table_name: str,
    ) -> DatasetMetadata:
        """Register an external SQLite or PostgreSQL table without copying data."""

        engine = self._get_engine(connection_url)
        inspector = inspect(engine)
        available_tables = inspector.get_table_names()
        if table_name not in available_tables:
            raise ValueError(
                f"Table '{table_name}' was not found in the provided database."
            )

        with engine.connect() as connection:
            preview_df = pd.read_sql_query(
                f"SELECT * FROM {table_name} LIMIT 5",
                connection,
            )
            count_df = pd.read_sql_query(
                f"SELECT COUNT(*) AS row_count FROM {table_name}",
                connection,
            )

        metadata = DatasetMetadata(
            name=sanitize_identifier(dataset_name),
            table_name=table_name,
            source_type="postgresql" if connection_url.startswith("postgres") else "sqlite",
            source_path=None,
            database_url=connection_url,
            row_count=int(count_df.iloc[0]["row_count"]),
            columns=[
                {"name": column["name"], "dtype": str(column["type"])}
                for column in inspector.get_columns(table_name)
            ],
            preview=preview_df.replace({pd.NA: None}).to_dict(orient="records"),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
        )
        self._registry[metadata.name] = metadata
        self._save_registry()
        logger.info(
            "Registered external dataset '%s' from table '%s'",
            metadata.name,
            table_name,
        )
        return metadata
