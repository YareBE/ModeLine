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
                st.session_state[key] = []
            elif key in ["processed_data", "description", "model", "na_method",
                         "df", "loaded_packet"]:
                st.session_state[key] = None

        st.session_state.file = uploaded_file
        if uploaded_file is None:
            return None

        extension = uploaded_file.name.split('.')[-1].lower()
        if extension != "joblib":
            df = _error_handler(uploaded_file, extension)
            with st.spinner("Loading data..."):
                if df.empty:
                    return None
                st.session_state.df = df
                st.success(
                    f"✅ Loaded {len(df)} rows, {len(df.columns)} columns"
                )
        else:
            with st.spinner("Loading data..."):
                try:
                    st.session_state.loaded_packet = joblib.load(uploaded_file)
                    st.session_state.model_name = (
                        uploaded_file.name.replace('.joblib', '')
                    )
                    st.success(f"✅ Model '{uploaded_file.name}' loaded.")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")


@st.cache_data(show_spinner=False)
def _error_handler(file, extension):
    """Load file based on extension and handle errors."""
    file.seek(0)
    conn = None

    try:
        if extension == 'csv':
            data = _upload_csv(file)
        elif extension in ('xls', 'xlsx'):
            data = _upload_excel(file)
        elif extension in ('db', 'sqlite'):
            conn = sqlite3.connect(':memory:')
            data = _upload_sql(file, conn)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        if data is None or data.empty:
            st.warning("Warning: Loaded file is empty")
        return data

    except Exception as err:
        st.error(f"Error while reading the data: {err}")
        return pd.DataFrame()

    finally:
        if extension in ('db', 'sqlite') and conn:
            conn.close()


def _upload_csv(file):
    """Read CSV file and normalize column names."""
    try:
        data = pd.read_csv(file)
        data.columns = data.columns.map(str)
        return data
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")


def _upload_excel(file):
    """Read Excel file and normalize column names."""
    try:
        data = pd.read_excel(file, engine='openpyxl')
        data.columns = data.columns.map(str)
        return data
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")


def _upload_sql(file, conn):
    """Read SQLite file and normalize column names."""
    try:
        conn.deserialize(file.read())
        table_query = (
            "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
        )
        tables = pd.read_sql_query(table_query, conn)
        if tables.empty:
            raise ValueError("No tables found in SQLite database")
        table = tables.iloc[0, 0]
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        data.columns = data.columns.map(str)
        return data
    except Exception as e:
        raise ValueError(f"Error reading SQLite file: {str(e)}")
