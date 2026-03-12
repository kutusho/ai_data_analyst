"""Authentication helpers for API endpoints."""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

bearer_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_token(
    request: Request,
    bearer_credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key: str | None = Security(api_key_scheme),
) -> None:
    """Require a shared API token when authentication is enabled."""

    expected_token = request.app.state.settings.api_auth_token
    if not expected_token:
        return

    provided_token = api_key
    if bearer_credentials and bearer_credentials.scheme.lower() == "bearer":
        provided_token = bearer_credentials.credentials

    if provided_token and secrets.compare_digest(provided_token, expected_token):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )
