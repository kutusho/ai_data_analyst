"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration for the platform."""

    app_name: str
    app_env: str
    api_host: str
    api_port: int
    api_base_url: str
    openai_api_key: str | None
    openai_model: str
    sqlite_path: Path
    default_database_url: str
    cache_dir: Path
    upload_dir: Path
    dataset_registry_path: Path
    history_path: Path
    artifacts_dir: Path
    charts_dir: Path
    reports_dir: Path
    datasets_dir: Path

    @property
    def openai_enabled(self) -> bool:
        """Return whether OpenAI integration is available."""

        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Create and cache the application configuration."""

    root_dir = Path(__file__).resolve().parent.parent
    cache_dir = root_dir / "cache"
    upload_dir = cache_dir / "uploads"
    artifacts_dir = root_dir / "artifacts"
    charts_dir = artifacts_dir / "charts"
    reports_dir = artifacts_dir / "reports"
    datasets_dir = root_dir / "datasets"
    sqlite_path = cache_dir / "analysis.db"

    for path in (
        cache_dir,
        upload_dir,
        charts_dir,
        reports_dir,
        datasets_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    default_database_url = os.getenv("DATABASE_URL", f"sqlite:///{sqlite_path}")

    return AppConfig(
        app_name=os.getenv("APP_NAME", "AI Data Analyst Platform"),
        app_env=os.getenv("APP_ENV", "development"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        sqlite_path=sqlite_path,
        default_database_url=default_database_url,
        cache_dir=cache_dir,
        upload_dir=upload_dir,
        dataset_registry_path=cache_dir / "dataset_registry.json",
        history_path=cache_dir / "history.json",
        artifacts_dir=artifacts_dir,
        charts_dir=charts_dir,
        reports_dir=reports_dir,
        datasets_dir=datasets_dir,
    )
