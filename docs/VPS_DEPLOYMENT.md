# VPS Deployment

## Overview

This project is intended to be deployed on a VPS with two long-running services:

- A FastAPI backend
- A Streamlit frontend

HTTPS should be handled by a reverse proxy such as Nginx or Traefik.

## Recommended Topology

```text
Internet
  -> Reverse proxy with HTTPS
  -> FastAPI service
  -> Streamlit service
  -> Optional PostgreSQL database
```

## Recommended Environment Variables

Backend:

```env
APP_ENV=production
API_HOST=0.0.0.0
API_PORT=8000
API_BASE_URL=https://your-api-domain.example
API_AUTH_TOKEN=replace-with-a-long-random-token
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=sqlite:///cache/analysis.db
```

Frontend:

```env
API_BASE_URL=https://your-api-domain.example
API_AUTH_TOKEN=replace-with-the-same-token
```

## Generic VPS Setup

Install Python and create a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Start the frontend:

```bash
streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

In production, both commands should be managed by `systemd`, Docker, or another process supervisor.

## Reverse Proxy

Expose the services through separate routes:

- `api.your-domain.example` -> FastAPI backend
- `app.your-domain.example` -> Streamlit frontend

The reverse proxy should:

- Terminate TLS
- Forward API traffic to the backend service
- Forward UI traffic to the Streamlit service
- Enforce upload size and timeout limits appropriate for CSV ingestion

## Verification

Basic health check:

```bash
curl https://your-api-domain.example/health
```

Example query:

```bash
curl -X POST https://your-api-domain.example/query \
  -H 'Content-Type: application/json' \
  -d '{"dataset_name":"tourism_data","question":"Which region has the highest tourism revenue?","options":{"forecast_periods":6}}'
```

## Security Notes

- Keep secrets in environment files or a secret manager, never in the repository
- Reuse the same `API_AUTH_TOKEN` between the backend and the frontend when the frontend calls the protected API
- Restrict direct public exposure of internal service ports when a reverse proxy is in front
- Prefer PostgreSQL over SQLite for heavier production workloads
- Rotate API keys and SSH credentials regularly
