"""Automated exploratory data analysis workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analytics.statistics import (
    correlation_matrix,
    dataset_profile,
    descriptive_statistics,
    missing_value_report,
    outlier_report,
    preferred_numeric_columns,
    seasonal_analysis,
    trend_analysis,
)


class EDAService:
    """Generate a reusable EDA summary for a DataFrame."""

    def run(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        """Return a full EDA package for the provided frame."""

        numeric_columns = preferred_numeric_columns(dataframe)
        categorical_columns = [
            column
            for column in dataframe.columns
            if column not in list(dataframe.select_dtypes(include=np.number).columns)
        ]
        return {
            "profile": dataset_profile(dataframe),
            "descriptive_statistics": descriptive_statistics(dataframe),
            "missing_values": missing_value_report(dataframe),
            "correlations": correlation_matrix(dataframe),
            "outliers": outlier_report(dataframe),
            "trend_analysis": trend_analysis(dataframe),
            "seasonal_analysis": seasonal_analysis(dataframe),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "distribution_candidates": numeric_columns[:3],
        }
