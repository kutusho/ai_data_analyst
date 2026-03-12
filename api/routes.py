"""HTTP routes for the AI Data Analyst Platform."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from api.controllers import DatasetUploadResponse, PlatformController, QueryRequest, QueryResponse
from api.security import require_api_token

router = APIRouter()


def get_controller(request: Request) -> PlatformController:
    """Resolve the shared platform controller from app state."""

    return request.app.state.controller


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query_data(
    payload: QueryRequest,
    _: None = Depends(require_api_token),
    controller: PlatformController = Depends(get_controller),
) -> QueryResponse:
    """Execute a natural language analytics request."""

    try:
        return controller.handle_query(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/upload-dataset", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile | None = File(default=None),
    dataset_name: str | None = Form(default=None),
    connection_url: str | None = Form(default=None),
    table_name: str | None = Form(default=None),
    _: None = Depends(require_api_token),
    controller: PlatformController = Depends(get_controller),
) -> DatasetUploadResponse:
    """Upload a CSV or register an external database table."""

    try:
        return await controller.handle_upload(
            file=file,
            dataset_name=dataset_name,
            connection_url=connection_url,
            table_name=table_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/insights")
async def get_insights(
    limit: int = 20,
    _: None = Depends(require_api_token),
    controller: PlatformController = Depends(get_controller),
) -> dict:
    """Return recent insights."""

    return controller.get_insights(limit=limit)


@router.get("/charts")
async def get_charts(
    limit: int = 20,
    _: None = Depends(require_api_token),
    controller: PlatformController = Depends(get_controller),
) -> dict:
    """Return recent chart metadata."""

    return controller.get_charts(limit=limit)


@router.get("/datasets")
async def list_datasets(
    _: None = Depends(require_api_token),
    controller: PlatformController = Depends(get_controller),
) -> dict:
    """Return available datasets."""

    return controller.list_datasets()


@router.get("/artifacts/{artifact_path:path}")
async def get_artifact(
    artifact_path: str,
    request: Request,
    _: None = Depends(require_api_token),
) -> FileResponse:
    """Serve chart and report artifacts behind the API token check."""

    artifacts_dir = request.app.state.settings.artifacts_dir.resolve()
    target_path = (artifacts_dir / artifact_path).resolve()

    try:
        target_path.relative_to(artifacts_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found.") from exc

    if not target_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")

    return FileResponse(path=Path(target_path))
