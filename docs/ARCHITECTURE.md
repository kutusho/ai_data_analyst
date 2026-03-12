# Architecture

## Overview

The platform is organized as a modular, layered analytics system:

1. `frontend/` provides the Streamlit user interface.
2. `api/` exposes HTTP endpoints through FastAPI.
3. `agents/` coordinates natural language analysis workflows.
4. `database/` provides universal dataset registration and safe query execution.
5. `analytics/`, `visualization/`, and `ml/` add domain-specific processing.
6. `artifacts/` and `cache/` persist outputs and lightweight operational state.

## Layer Breakdown

### User Interface Layer

File: `frontend/streamlit_app.py`

Responsibilities:

- Upload CSV datasets
- Register external database tables
- Submit natural language questions
- Render analysis, SQL, insights, and recommendations
- Display downloadable chart and PDF report artifacts
- Show recent insight and chart history

## API Layer

Files:

- `backend/main.py`
- `api/routes.py`
- `api/controllers.py`

Responsibilities:

- Initialize the application and shared controller state
- Validate incoming requests with Pydantic models
- Route HTTP requests to the controller
- Return structured JSON responses
- Serve static artifacts under `/artifacts`

Current endpoints:

- `GET /health`
- `GET /datasets`
- `POST /query`
- `POST /upload-dataset`
- `GET /insights`
- `GET /charts`

## Agent Layer

Files:

- `agents/orchestrator.py`
- `agents/sql_agent.py`
- `agents/analysis_agent.py`
- `agents/visualization_agent.py`
- `agents/insight_agent.py`

### Orchestrator Agent

The orchestrator is the entry point for decision-making. It classifies each question into one of four workflows:

- `sql_analysis`
- `forecast`
- `cluster`
- `anomaly`

It then coordinates the relevant agents and services and normalizes the response payload returned to the API layer.

### SQL Generation Agent

Responsibilities:

- Translate natural language questions to SQL
- Infer dimensions and metrics from dataset columns
- Generate read-only `SELECT` statements
- Prevent destructive SQL patterns

### Data Analysis Agent

Responsibilities:

- Summarize query outputs
- Infer metric columns
- Generate descriptive statistics and evidence
- Summarize forecast, cluster, and anomaly outputs

### Visualization Agent

Responsibilities:

- Infer chart type from the question and result frame
- Generate Plotly HTML artifacts
- Support line, bar, scatter, histogram, and specialized ML visualizations

### Insight Agent

Responsibilities:

- Turn analysis evidence into human-readable findings
- Produce recommendations
- Return explainability traces
- Use OpenAI when configured, otherwise fall back to deterministic summaries

## Data Layer

Files:

- `database/connector.py`
- `database/query_executor.py`

### Universal Data Connector

Supported sources:

- CSV files
- SQLite databases
- PostgreSQL tables

Capabilities:

- Dataset registration
- Dataset metadata and schema lookup
- Automatic loading of bundled local datasets
- Dataset registry persistence

### Safe Query Executor

Responsibilities:

- Execute only validated read-only SQL
- Cache repeated queries
- Convert results into JSON-safe previews
- Load raw dataset frames for EDA and ML workflows

## Analytics Layer

Files:

- `analytics/eda.py`
- `analytics/statistics.py`
- `analytics/forecasting.py`

Capabilities:

- Descriptive statistics
- Missing value detection
- Correlation analysis
- Outlier reporting
- Trend analysis
- Seasonal summaries
- Dataset profiling

## Machine Learning Layer

Files:

- `ml/models.py`
- `ml/training.py`

Supported workflows:

- Linear trend projection for forecasts
- Time-series preparation and projection
- Clustering
- Anomaly detection

These workflows are triggered directly by the orchestrator based on intent keywords in the question.

## Artifact and State Storage

Runtime directories:

- `cache/dataset_registry.json`
- `cache/history.json`
- `cache/uploads/`
- `artifacts/charts/`
- `artifacts/reports/`

Stored outputs:

- Dataset registry metadata
- Recent insight history
- Recent chart history
- HTML chart exports
- PDF reports
- Uploaded CSV files

## Request Lifecycle

### SQL analysis path

1. Streamlit sends a `POST /query`.
2. FastAPI validates the payload.
3. The controller invokes the orchestrator.
4. The SQL agent builds safe SQL.
5. The query executor runs the query with caching.
6. The analysis agent summarizes the result.
7. The visualization agent writes a chart artifact.
8. The insight agent produces findings and recommendations.
9. The API returns a structured response to the UI.

### Forecast path

1. The orchestrator detects forecast intent.
2. The full dataset is loaded into a pandas DataFrame.
3. The target metric is inferred from the question.
4. The ML service projects future values.
5. Analysis, charting, and insights are generated from the forecast output.

## Extensibility

The codebase is designed so new capabilities can be added with small, isolated changes:

- New agents can be introduced and wired into the orchestrator
- New data sources can be added in `database/connector.py`
- New chart types can be added in `visualization/charts.py`
- Additional ML workflows can be added in `ml/training.py`
- New API endpoints can be mounted in `api/routes.py`

## Operational Characteristics

- Read-only SQL guardrails are enforced before execution
- Artifacts are served statically by FastAPI
- OpenAI support is optional and disabled by default when no API key is present
- The frontend remains functional without OpenAI because deterministic fallbacks are implemented in the agent layer
