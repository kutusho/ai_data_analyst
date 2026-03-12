"""Statistical helpers for automated exploratory data analysis."""

from __future__ import annotations

import calendar
from typing import Any

import numpy as np
import pandas as pd


def preferred_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return numeric columns that are likely to be analytical metrics."""

    excluded = {
        "year",
        "month",
        "day",
        "week",
        "quarter",
        "cluster",
        "anomaly_flag",
    }
    numeric_columns = list(dataframe.select_dtypes(include=np.number).columns)
    preferred = [
        column
        for column in numeric_columns
        if column.lower() not in excluded and not column.lower().endswith("_id")
    ]
    return preferred or numeric_columns


def _month_to_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and 1 <= int(value) <= 12:
        return int(value)

    normalized = str(value).strip().lower()
    if normalized.isdigit() and 1 <= int(normalized) <= 12:
        return int(normalized)

    month_map = {
        name.lower(): index
        for index, name in enumerate(calendar.month_name)
        if name
    }
    month_map.update(
        {
            name.lower(): index
            for index, name in enumerate(calendar.month_abbr)
            if name
        }
    )
    return month_map.get(normalized)


def build_time_index(dataframe: pd.DataFrame) -> pd.Series | None:
    """Infer a time axis from the available columns."""

    if dataframe.empty:
        return None

    for column in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            return pd.to_datetime(dataframe[column], errors="coerce")

    time_candidates = [
        column
        for column in dataframe.columns
        if any(token in column.lower() for token in ("date", "time", "period"))
    ]
    if time_candidates:
        series = pd.to_datetime(dataframe[time_candidates[0]], errors="coerce")
        if series.notna().any():
            return series

    lowered_columns = {column.lower(): column for column in dataframe.columns}
    if {"year", "month"}.issubset(lowered_columns):
        year_col = lowered_columns["year"]
        month_col = lowered_columns["month"]
        years = pd.to_numeric(dataframe[year_col], errors="coerce")
        months = dataframe[month_col].map(_month_to_int)
        if years.notna().any() and months.notna().any():
            return pd.to_datetime(
                {
                    "year": years.ffill().astype(int),
                    "month": months.fillna(1).astype(int),
                    "day": 1,
                },
                errors="coerce",
            )

    if "year" in lowered_columns:
        year_series = pd.to_numeric(dataframe[lowered_columns["year"]], errors="coerce")
        if year_series.notna().any():
            return pd.to_datetime(year_series.astype("Int64").astype(str), format="%Y", errors="coerce")

    return None


def dataset_profile(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build a compact dataset profile."""

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": [
            {
                "name": column,
                "dtype": str(dtype),
                "non_null": int(dataframe[column].notna().sum()),
                "nulls": int(dataframe[column].isna().sum()),
                "unique_values": int(dataframe[column].nunique(dropna=True)),
            }
            for column, dtype in dataframe.dtypes.items()
        ],
    }


def descriptive_statistics(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Compute descriptive statistics for numeric columns."""

    numeric_df = dataframe[preferred_numeric_columns(dataframe)]
    if numeric_df.empty:
        return {}
    return numeric_df.describe().round(4).to_dict()


def missing_value_report(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Summarize missing values by column."""

    total_rows = max(len(dataframe), 1)
    return {
        column: {
            "missing_count": int(dataframe[column].isna().sum()),
            "missing_pct": round(float(dataframe[column].isna().mean() * 100), 2),
        }
        for column in dataframe.columns
        if dataframe[column].isna().sum() > 0 or total_rows > 0
    }


def correlation_matrix(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a numeric correlation matrix."""

    numeric_columns = preferred_numeric_columns(dataframe)
    numeric_df = dataframe[numeric_columns] if numeric_columns else dataframe.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return {}
    return numeric_df.corr(numeric_only=True).round(4).to_dict()


def outlier_report(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Detect outliers using the IQR method."""

    report: dict[str, Any] = {}
    numeric_columns = preferred_numeric_columns(dataframe)
    numeric_df = dataframe[numeric_columns] if numeric_columns else dataframe.select_dtypes(include=np.number)
    for column in numeric_df.columns:
        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            report[column] = {"outlier_count": 0, "lower_bound": q1, "upper_bound": q3}
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)
        report[column] = {
            "outlier_count": int(mask.sum()),
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
        }
    return report


def trend_analysis(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Detect simple trends over time for the first numeric metric."""

    time_index = build_time_index(dataframe)
    numeric_columns = preferred_numeric_columns(dataframe)
    if time_index is None or not numeric_columns:
        return {}

    metric = numeric_columns[0]
    trend_df = pd.DataFrame({"period": time_index, metric: dataframe[metric]}).dropna()
    if trend_df.empty:
        return {}

    trend_df = trend_df.groupby("period", as_index=False)[metric].sum().sort_values("period")
    first_value = float(trend_df[metric].iloc[0])
    last_value = float(trend_df[metric].iloc[-1])
    absolute_change = last_value - first_value
    percent_change = (absolute_change / first_value * 100) if first_value else None
    return {
        "metric": metric,
        "first_period": trend_df["period"].iloc[0].strftime("%Y-%m-%d"),
        "last_period": trend_df["period"].iloc[-1].strftime("%Y-%m-%d"),
        "first_value": round(first_value, 4),
        "last_value": round(last_value, 4),
        "absolute_change": round(float(absolute_change), 4),
        "percent_change": round(float(percent_change), 2) if percent_change is not None else None,
    }


def seasonal_analysis(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Group metrics by month when a month column is available."""

    lowered_columns = {column.lower(): column for column in dataframe.columns}
    if "month" not in lowered_columns:
        return {}

    month_col = lowered_columns["month"]
    numeric_columns = preferred_numeric_columns(dataframe)
    if not numeric_columns:
        return {}

    metric = numeric_columns[0]
    season_df = dataframe[[month_col, metric]].copy()
    season_df["month_number"] = season_df[month_col].map(_month_to_int)
    season_df = season_df.dropna(subset=["month_number"])
    if season_df.empty:
        return {}

    grouped = (
        season_df.groupby("month_number")[metric]
        .mean()
        .round(4)
        .sort_index()
        .to_dict()
    )
    return {
        "metric": metric,
        "monthly_average": {
            calendar.month_abbr[int(month)]: float(value)
            for month, value in grouped.items()
        },
    }
