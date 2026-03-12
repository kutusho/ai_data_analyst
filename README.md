# AI Data Analyst Platform

Production-oriented natural language analytics platform built with FastAPI, Streamlit, pandas, SQLAlchemy, Plotly, and scikit-learn.

## Features

- Natural language to SQL with safe read-only validation
- CSV upload plus SQLite and PostgreSQL table registration
- Automated EDA with profiling, missing values, correlations, outliers, trend, and seasonal summaries
- Interactive Plotly visualizations with chart artifact export
- Insight generation with optional OpenAI-backed summarization
- Predictive workflows for trend projection, clustering, and anomaly detection
- Query caching and PDF report export

## Project Structure

```text
backend/
api/
agents/
analytics/
database/
frontend/
ml/
utils/
visualization/
datasets/
artifacts/
cache/
```

## Setup

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and optionally add an OpenAI API key:

```bash
cp .env.example .env
```

## Run

Backend:

```bash
uvicorn backend.main:app --reload
```

Frontend:

```bash
streamlit run frontend/streamlit_app.py
```

## API Endpoints

- `POST /query`
- `POST /upload-dataset`
- `GET /insights`
- `GET /charts`
- `GET /datasets`
- `GET /health`

## Example Request

```json
{
  "dataset_name": "tourism_data",
  "question": "Which region has the highest tourism revenue?",
  "options": {
    "forecast_periods": 12
  }
}
```

## Example Response

```json
{
  "analysis": "The query returned 3 rows and 2 columns...",
  "insights": "Tourism revenue is concentrated in a single region...",
  "chart_url": "/artifacts/charts/tourism_data_chart.html",
  "recommendations": "Double down on the leading segment..."
}
```

## Notes

- OpenAI integration is optional. Without `OPENAI_API_KEY`, the platform uses deterministic fallback logic.
- Bundled sample datasets are auto-registered at startup.
- Generated charts and reports are stored under `artifacts/`.
