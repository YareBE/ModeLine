import pandas as pd
import sqlite3
import openpyxl
import streamlit as st
import joblib


def upload_file():
    """Load dataset or previously saved model from file."""
    uploaded_file = st.file_uploader(
        "Upload your dataset or a previously saved model",
        type=["csv", "xls", "xlsx", "db", "sqlite", "joblib"],
        help="Supported formats: CSV, Excel, SQLite and Joblib",
        key="current_file"
    )

    # Only reset if file actually changed (by name and size)
    file_changed = False
    if uploaded_file is not None and st.session_state.file is not None:
        # File changed if name or size is different
        file_changed = (
            uploaded_file.name != st.session_state.file.name or
            uploaded_file.size != st.session_state.file.size
        )
    elif uploaded_file is None and st.session_state.file is not None:
        # File was cleared
        file_changed = True
    elif uploaded_file is not None and st.session_state.file is None:
        # New file selected (first time)
        file_changed = True

    if file_changed:
        # Reset session state only when file actually changes
        for key in st.session_state:
            if key == "features" or key == "target":
                # Reset selection lists to empty
                st.session_state[key] = []
            elif key in ["processed_data", "description", "model", "na_method",
                         "df", "loaded_packet"]:
                # Reset objects to None
                st.session_state[key] = None
        
        # Store uploaded file reference
        st.session_state.file = uploaded_file
        if uploaded_file is None:
            return None
        
        # Extract file extension for format detection
        extension = uploaded_file.name.split('.')[-1].lower()
        if extension != "joblib":
            # Handle datafile (CSV, Excel and SQLite)
            df = _error_handler(uploaded_file, extension)
            with st.spinner("Loading data..."):
                # Validate DataFrame is not empty
                if df.empty:
                    return None
                # Store DataFrame in session state
                st.session_state.df = df
                st.success(
                    f"✅ Loaded {len(df)} rows, {len(df.columns)} columns"
                )
        else:
            # Handle model files (Joblib)
            with st.spinner("Loading data..."):
                try:
                    # Load serialized model packet
                    st.session_state.loaded_packet = joblib.load(uploaded_file)
                    # Store model name without extension
                    st.session_state.model_name = (
                        uploaded_file.name.replace('.joblib', '')
                    )
                    st.success(f"✅ Model '{uploaded_file.name}' loaded.")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")


@st.cache_data(show_spinner=False)
def _error_handler(file, extension):
    """Load file based on extension and handle errors."""
    # Reset file pointer to ensure full file is read
    file.seek(0)
    conn = None
    
    try:
        # Route to appropriate loader based on extension
        if extension == 'csv':
            data = _upload_csv(file)
        elif extension in ('xls', 'xlsx'):
            data = _upload_excel(file)
        elif extension in ('db', 'sqlite'):
            # SQLite requires connection object for querying
            conn = sqlite3.connect(':memory:')
            data = _upload_sql(file, conn)
        else:
            # Unsupported extension (shouldn't happen due to uploader filter)
            raise ValueError(f"Unsupported file extension: {extension}")
        
        # Validate loaded data is not empty
        if data is None or data.empty:
            st.warning("Warning: Loaded file is empty")
        return data

    except Exception as err:
        # Catch all errors and display user-friendly message
        st.error(f"Error while reading the data: {err}")
        # Return empty DataFrame instead of None for consistent error handling
        return pd.DataFrame()

    finally:
        # Always close SQLite connection to prevent resource leaks
        if extension in ('db', 'sqlite') and conn:
            conn.close()


def _upload_csv(file):
    """Read CSV file and normalize column names."""
    try:
        data = pd.read_csv(file)
        # Convert all column names to strings (handles numeric/mixed types)
        data.columns = data.columns.map(str)
        return data
    except pd.errors.EmptyDataError:
        # Specific error for empty CSV files
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError as e:
        # Parsing errors (malformed CSV, encoding issues, etc.)
        raise ValueError(f"Error parsing CSV: {str(e)}")


def _upload_excel(file):
    """Read Excel file and normalize column names."""
    try:
        # Use openpyxl engine for modern Excel format support
        data = pd.read_excel(file, engine='openpyxl')
        # Convert all column names to strings
        data.columns = data.columns.map(str)
        return data
    except Exception as e:
        # Catch all Excel-related errors (corrupted file, wrong format, etc.)
        raise ValueError(f"Error reading Excel file: {str(e)}")


def _upload_sql(file, conn):
    """Read SQLite file and normalize column names."""
    try:
        # Load SQLite file content into in-memory database
        conn.deserialize(file.read())

        # Query for available tables in database
        table_query = (
            "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
        )
        tables = pd.read_sql_query(table_query, conn)

        # Validate that at least one table exists 
        if tables.empty:
            raise ValueError("No tables found in SQLite database")
        # Get name of first table
        table = tables.iloc[0, 0]
        # Read entire table into DataFrame
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        # Convert all column names to strings
        data.columns = data.columns.map(str)
        return data
    except Exception as e:
        # Catch all SQLite related errors
        raise ValueError(f"Error reading SQLite file: {str(e)}")

