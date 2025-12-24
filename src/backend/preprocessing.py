"""Data preprocessing utilities for feature and target selection.

This module provides functions for loading data, identifying numeric columns,
detecting and handling missing values (NAs), and configuring train/test splits.
All functions include type hints and comprehensive docstrings.
"""

import streamlit as st
from typing import Optional, Union, List
import pandas as pd


@st.cache_data(show_spinner=False)
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of numeric column names from a DataFrame.

    Inspects a DataFrame and returns column names containing numeric data types
    (int, float, etc.). Excludes columns with object, string, or categorical types.

    Args:
        df (pd.DataFrame): Input DataFrame to inspect.

    Returns:
        List[str]: Column names of numeric dtype.
    """
    return df.select_dtypes(include=['number']).columns.tolist()


@st.cache_data(show_spinner=False)
def get_na_info(df: pd.DataFrame) -> List[str]:
    """Return a list of columns that contain missing values.

    Scans the DataFrame to identify any columns with NaN/None values
    and returns their names. Useful for deciding NA handling strategy.

    Args:
        df (pd.DataFrame): Input DataFrame to inspect.

    Returns:
        List[str]: Column names that have at least one NA/NaN value.
    """
    cols_with_na = df.columns[df.isna().any()].tolist()
    return cols_with_na


@st.cache_data(show_spinner=False)
def apply_na_handling(
    df: pd.DataFrame,
    method: str,
    constant_value: Optional[Union[int, float]] = None
) -> pd.DataFrame:
    """Apply a missing-value handling strategy to a DataFrame.

    Implements multiple strategies for handling NAs including row deletion,
    mean/median imputation for numeric columns, or constant value filling.
    Returns a copy of the DataFrame; original is not modified.

    Args:
        df (pd.DataFrame): DataFrame to process (a copy is processed).
        method (str): Strategy to apply. Must be one of:
            - "Delete rows": Remove all rows containing any NA value
            - "Mean": Fill NAs in numeric columns with their mean
            - "Median": Fill NAs in numeric columns with their median
            - "Constant": Fill all NAs with the provided constant value
        constant_value (Optional[Union[int, float]]): Value to use when
            method == "Constant". Will be converted to float. Ignored for other methods.

    Returns:
        pd.DataFrame: Processed DataFrame after applying NA handling strategy.

    Raises:
        ValueError: If method == "Constant" but constant_value is not numeric.
        RuntimeError: If an unexpected error occurs during processing.
    """
    df = df.copy()
    if method == "Constant" and constant_value is not None:
        try:
            # Convert constant to float and fill all NAs
            value = float(constant_value)
            return df.fillna(value)
        except (ValueError, TypeError):
            raise ValueError("Constant value must be numeric")
    try:
        if method == "Delete rows":
            # Delete all rows that contain at least one NA value
            return df.dropna()
        elif method == "Mean":
            # Fill NAs in numeric columns with mean value
            return df.fillna(df.mean(numeric_only=True))
        elif method == "Median":
            # Fill NAs in numeric columns with median value
            return df.fillna(df.median(numeric_only=True))
        # If method not recognized, return original copy unchanged
        return df
    except Exception as e:
        raise RuntimeError(f"Error applying NA handling: {str(e)}")
