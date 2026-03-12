"""FastAPI entrypoint for the AI Data Analyst Platform."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.controllers import PlatformController
from api.routes import router
from backend.config import get_settings


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Natural language data analysis platform powered by specialized agents.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.controller = PlatformController(settings)
    app.include_router(router)
    app.mount("/artifacts", StaticFiles(directory=settings.artifacts_dir), name="artifacts")

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "message": settings.app_name,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
