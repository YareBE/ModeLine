"""Data uploading and parsing utilities for various file formats.

This module handles loading data from multiple file formats:
- CSV and Excel (.xlsx) files for tabular data
- SQLite databases for structured data
- joblib files containing serialized ModeLine model packets

All functions include type hints, comprehensive docstrings, and robust error handling.
"""

import pandas as pd
import sqlite3
import joblib
from typing import BinaryIO, Dict, Any


class InvalidJoblibPacket(Exception):
    """Raised when a joblib file is not a valid ModeLine model packet."""
    pass


def dataset_error_handler(file: BinaryIO, extension: str) -> pd.DataFrame:
    """Dispatch file reading based on extension and handle common errors.

    This function reads the uploaded file using the appropriate helper
    depending on the detected extension. For SQLite files, an in-memory
    SQLite connection is used to deserialize the content.

    Args:
        file (BinaryIO): Uploaded file-like object (Streamlit upload provides this).
        extension (str): File extension token (e.g. 'csv', 'xlsx', 'db').

    Returns:
        pd.DataFrame: Loaded DataFrame (may be empty on read errors).

    Raises:
        ValueError: When an unsupported extension is passed.
    """
    # Reset file pointer to ensure full file is read
    file.seek(0)
    if extension not in ['csv', 'xls', 'xlsx', 'db', 'sqlite']:
        raise ValueError(f"Unsupported file extension: {extension}")

    try:
        # Route to appropriate loader based on extension
        if extension == 'csv':
            data = _upload_csv(file)
        elif extension in ('xls', 'xlsx'):
            data = _upload_excel(file)
        elif extension in ('db', 'sqlite'):
            # SQLite requires in-memory connection for querying
            data = _upload_sql(file)
    except Exception as e:  # Fallback for unexpected errors
        raise RuntimeError(f"An unexpected error occurred while " \
                           f"processing the file: {str(e)}")

    else:
        # Validate loaded data is not empty
        if data.empty:
            raise pd.errors.EmptyDataError(f"Empty {extension}")
        return data


def _upload_csv(file: BinaryIO) -> pd.DataFrame:
    """Read a CSV file into a DataFrame and normalize its column names.

    Handles common encoding issues (UTF-8 and latin-1) and provides
    descriptive error messages for malformed CSV data.

    Args:
        file (BinaryIO): File-like object positioned at start.

    Returns:
        pd.DataFrame: Parsed CSV content with stringified column names.

    Raises:
        ValueError: If the CSV is empty or cannot be parsed.
    """
    try:
        data = pd.read_csv(file)

    except UnicodeDecodeError:
        # pd.read_csv expects UTF-8 by default, fall back to latin-1
        file.seek(0)
        try:
            data = pd.read_csv(file, encoding='latin-1')
        except UnicodeDecodeError:
            raise UnicodeDecodeError(
                "utf-8",
                b"",
                0,
                1,
                "Error while decoding the csv. ModeLine expects UTF-8 or latin-1")

    except pd.errors.EmptyDataError:
        # Specific error for empty CSV files
        raise pd.errors.EmptyDataError("CSV file is empty")

    except pd.errors.ParserError as e:
        # Parsing errors (malformed CSV, encoding issues, etc.)
        raise pd.errors.ParserError(f"Error parsing CSV: {str(e)}")

    except Exception as e:
        raise ValueError(f"Error reading csv file: {str(e)}")

    else:
        # Convert all column names to strings (handles numeric/mixed types)
        data.columns = data.columns.map(str)
        return data


def _upload_excel(file: BinaryIO) -> pd.DataFrame:
    """Read an Excel file into a DataFrame using openpyxl engine.

    Loads the first sheet of an Excel workbook and converts column names
    to strings for consistency with other data loaders.

    Args:
        file (BinaryIO): File-like object representing the Excel workbook.

    Returns:
        pd.DataFrame: Parsed first sheet with string column names.

    Raises:
        ValueError: If there is any problem reading the Excel file.
    """
    try:
        # Use openpyxl engine for modern Excel format (.xlsx) support
        data = pd.read_excel(file, engine='openpyxl')

    except pd.errors.EmptyDataError:
        # Specific error for empty Excel files
        raise pd.errors.EmptyDataError("Excel file is empty")

    except Exception as e:
        # Catch all Excel-related errors (corrupted file, wrong format, etc.)
        raise ValueError(f"Error reading Excel file: {str(e)}")

    else:
        # Convert all column names to strings
        data.columns = data.columns.map(str)
        return data


def _upload_sql(file: BinaryIO) -> pd.DataFrame:
    """Load a SQLite database file into a DataFrame by reading the first table.

    Creates an in-memory SQLite connection, deserializes the uploaded database
    file into it, and reads the first available table into a pandas DataFrame.

    Args:
        file (BinaryIO): Uploaded SQLite file-like object.

    Returns:
        pd.DataFrame: Contents of the first table in the database.

    Raises:
        ValueError: If the database contains no tables or cannot be read.
    """
    try:
        conn = sqlite3.connect(':memory:')
        # Load SQLite file content into in-memory database
        conn.deserialize(file.read())

        # Query for available tables in database
        table_query = (
            "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
        )
        tables = pd.read_sql_query(table_query, conn)

        # Validate that at least one table exists
        if tables.empty:
            raise pd.errors.EmptyDataError(
                "No tables found in SQLite database")

        # Get name of first table
        table = tables.iloc[0, 0]
        # Read entire table into DataFrame
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    except Exception as e:
        # Catch all SQLite related errors
        raise ValueError(f"Error reading SQLite file: {str(e)}")

    else:
        # Convert all column names to strings
        data.columns = data.columns.map(str)
        return data

    finally:
        conn.close()


def upload_joblib(file: BinaryIO) -> Dict[str, Any]:
    """Load a joblib-serialized ModeLine model packet from file.

    Deserializes a joblib binary file and validates it contains a ModeLine
    model packet (checks for 'app' key with value 'ModeLine'). Returns the
    packet dictionary containing model, features, target, metrics, etc.

    Args:
        file (BinaryIO): Uploaded joblib file-like object.

    Returns:
        Dict[str, Any]: Packet dictionary with keys:
            - 'model': LinearRegression model
            - 'features': List of feature names
            - 'target': List with target variable name
            - 'formula': Formula string
            - 'metrics': Dict with performance metrics
            - 'description': User description
            - 'app': Must be 'ModeLine'

    Raises:
        InvalidJoblibPacket: If joblib file is not from ModeLine.
        ValueError: If joblib file cannot be read or deserialized.
    """
    try:
        packet = joblib.load(file)

    except Exception as e:
        raise ValueError(f"Unexpected error reading joblib file: {str(e)}")

    else:
        # Validate this is a ModeLine packet
        if not packet.get("app") or packet.get("app") != 'ModeLine':
            raise InvalidJoblibPacket("This joblib file is not from ModeLine!")
        return packet
