"""Streamlit frontend for the AI Data Analyst Platform."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AI Data Analyst Platform",
    page_icon="📊",
    layout="wide",
)


def api_url(base_url: str, path: str) -> str:
    """Build a full API URL from the base URL and path."""

    return f"{base_url.rstrip('/')}{path}"


def request_headers(api_token: str | None) -> dict[str, str]:
    """Build authenticated request headers when a token is configured."""

    token = (api_token or "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


@st.cache_data(ttl=15)
def fetch_json(base_url: str, path: str, api_token: str | None) -> dict[str, Any]:
    """Load JSON data from the API."""

    response = requests.get(
        api_url(base_url, path),
        headers=request_headers(api_token),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def post_query(base_url: str, payload: dict[str, Any], api_token: str | None) -> dict[str, Any]:
    """Send a query request to the API."""

    response = requests.post(
        api_url(base_url, "/query"),
        json=payload,
        headers=request_headers(api_token),
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def post_upload(
    base_url: str,
    files: dict[str, Any] | None,
    data: dict[str, Any],
    api_token: str | None,
) -> dict[str, Any]:
    """Send an upload request to the API."""

    response = requests.post(
        api_url(base_url, "/upload-dataset"),
        files=files,
        data=data,
        headers=request_headers(api_token),
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def fetch_artifact_text(base_url: str, path: str, api_token: str | None) -> str:
    """Fetch an artifact as text."""

    response = requests.get(
        api_url(base_url, path),
        headers=request_headers(api_token),
        timeout=60,
    )
    response.raise_for_status()
    return response.text


def fetch_artifact_bytes(base_url: str, path: str, api_token: str | None) -> bytes:
    """Fetch an artifact as bytes."""

    response = requests.get(
        api_url(base_url, path),
        headers=request_headers(api_token),
        timeout=60,
    )
    response.raise_for_status()
    return response.content


st.title("AI Data Analyst Platform")
st.caption("Natural language analytics, SQL generation, visualizations, and predictive workflows.")

default_api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
default_api_token = os.getenv("API_AUTH_TOKEN", "")
api_base = st.sidebar.text_input("FastAPI URL", value=default_api_url)
manual_api_token = st.sidebar.text_input("API token", value="", type="password")
api_token = manual_api_token.strip() or default_api_token
if default_api_token:
    st.sidebar.caption("Server-managed API token is configured.")
forecast_periods = st.sidebar.number_input("Forecast periods", min_value=1, max_value=60, value=12)

st.sidebar.subheader("CSV Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
upload_name = st.sidebar.text_input("Dataset name", value="custom_dataset")
if st.sidebar.button("Register CSV Dataset", use_container_width=True):
    if uploaded_file is None:
        st.sidebar.error("Choose a CSV file first.")
    else:
        try:
            upload_result = post_upload(
                api_base,
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                data={"dataset_name": upload_name},
                api_token=api_token,
            )
            st.session_state["latest_upload"] = upload_result
            fetch_json.clear()
            st.sidebar.success(upload_result["message"])
        except requests.RequestException as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.sidebar.error(f"Upload failed: {detail}")

st.sidebar.subheader("Database Registration")
external_dataset_name = st.sidebar.text_input("External dataset name", value="external_dataset")
connection_url = st.sidebar.text_input("Connection URL", placeholder="postgresql+psycopg://user:pass@host/db")
table_name = st.sidebar.text_input("Table name", placeholder="tourism_data")
if st.sidebar.button("Register Database Table", use_container_width=True):
    try:
        upload_result = post_upload(
            api_base,
            files=None,
            data={
                "dataset_name": external_dataset_name,
                "connection_url": connection_url,
                "table_name": table_name,
            },
            api_token=api_token,
        )
        st.session_state["latest_upload"] = upload_result
        fetch_json.clear()
        st.sidebar.success(upload_result["message"])
    except requests.RequestException as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.sidebar.error(f"Registration failed: {detail}")

try:
    datasets_payload = fetch_json(api_base, "/datasets", api_token)
    dataset_items = datasets_payload.get("items", [])
    dataset_names = [item["name"] for item in dataset_items]
except requests.RequestException as exc:
    dataset_items = []
    dataset_names = []
    st.error(f"Could not reach the backend: {exc}")

left, right = st.columns([2, 1])

with left:
    selected_dataset = st.selectbox(
        "Dataset",
        options=dataset_names or ["tourism_data"],
        index=0,
    )
    question = st.text_area(
        "Ask a question about your data",
        value="Which region has the highest tourism revenue?",
        height=120,
    )
    run_query = st.button("Run Analysis", type="primary", use_container_width=True)

with right:
    st.subheader("Example Prompts")
    st.markdown(
        "\n".join(
            [
                "- Which region has the highest tourism revenue?",
                "- Show the monthly growth of visitors.",
                "- Predict next year's tourism demand.",
                "- Detect anomalies in tourism revenue.",
                "- Cluster regions based on visitors and revenue.",
            ]
        )
    )

if run_query and question.strip():
    try:
        result = post_query(
            api_base,
            {
                "dataset_name": selected_dataset,
                "question": question,
                "options": {"forecast_periods": int(forecast_periods)},
            },
            api_token,
        )
        st.session_state["latest_result"] = result
        fetch_json.clear()
    except requests.RequestException as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Query failed: {detail}")

latest_result = st.session_state.get("latest_result")
if latest_result:
    analysis_col, insight_col = st.columns(2)
    analysis_col.subheader("Analysis")
    analysis_col.write(latest_result["analysis"])
    insight_col.subheader("Insights")
    insight_col.write(latest_result["insights"])

    st.subheader("Recommendations")
    st.write(latest_result["recommendations"])

    if latest_result.get("sql"):
        st.subheader("Generated SQL")
        st.code(latest_result["sql"], language="sql")

    details_tab, data_tab, trace_tab = st.tabs(["Chart", "Data Preview", "Explainability"])
    with details_tab:
        chart_url = latest_result.get("chart_url")
        if chart_url:
            html = fetch_artifact_text(api_base, chart_url, api_token)
            components.html(html, height=560, scrolling=True)
            st.download_button(
                "Download chart HTML",
                data=html.encode("utf-8"),
                file_name=f"{latest_result['workflow']}_chart.html",
                mime="text/html",
                use_container_width=True,
            )
        else:
            st.info("No chart was generated for this result.")

        if latest_result.get("report_url"):
            pdf_bytes = fetch_artifact_bytes(api_base, latest_result["report_url"], api_token)
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name="analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with data_tab:
        st.dataframe(latest_result.get("data_preview", []), use_container_width=True)

    with trace_tab:
        st.write(latest_result.get("explainability", []))
        st.json(latest_result.get("analysis_details", {}))

latest_upload = st.session_state.get("latest_upload")
if latest_upload:
    st.subheader("Latest Dataset Profile")
    st.write(latest_upload["message"])
    st.json(latest_upload["profile"])
    if latest_upload.get("charts"):
        for chart in latest_upload["charts"]:
            chart_html = fetch_artifact_text(api_base, chart["url"], api_token)
            components.html(chart_html, height=420, scrolling=True)

history_col, chart_col = st.columns(2)
with history_col:
    st.subheader("Recent Insights")
    try:
        insights_payload = fetch_json(api_base, "/insights", api_token)
        st.dataframe(insights_payload.get("items", []), use_container_width=True)
    except requests.RequestException:
        st.info("Insight history is unavailable.")

with chart_col:
    st.subheader("Recent Charts")
    try:
        charts_payload = fetch_json(api_base, "/charts", api_token)
        st.dataframe(charts_payload.get("items", []), use_container_width=True)
    except requests.RequestException:
        st.info("Chart history is unavailable.")
