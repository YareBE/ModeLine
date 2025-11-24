import pandas as pd
import sqlite3
import streamlit as st
import joblib


def upload_file():
    """Load a dataset or a previously saved model from an uploaded file.

    This helper is intended to be used within a Streamlit app. The function
    renders a ``st.file_uploader`` control and loads the provided file based
    on its extension. Supported formats include CSV, Excel, SQLite databases
    and serialized Joblib packets containing a saved model.

    Side effects:
        - Updates keys in ``st.session_state`` such as ``df``,
          ``loaded_packet``, ``model_name`` and ``file``.

    Returns:
        None: Results are stored into ``st.session_state`` for downstream use.
    """
    uploaded_file = st.file_uploader(
        "Upload your dataset or a previously saved model",
        type=["csv", "xls", "xlsx", "db", "sqlite", "joblib"],
        help="Supported formats: CSV, Excel, SQLite and Joblib"
    )

    if st.session_state.file != uploaded_file:
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
                st.success("✅ Dataset correctly loaded.")
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
                    st.success("✅ Model correctly loaded.")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")


@st.cache_data(show_spinner=False)
def _error_handler(file, extension):
    """Dispatch file reading based on extension and handle common errors.

    This function reads the uploaded ``file`` using the appropriate helper
    depending on the detected ``extension``. For SQLite files, an in-memory
    SQLite connection is used to deserialize the content.

    Args:
        file (io.BufferedIOBase): Uploaded file-like object (Streamlit
            upload provides this).
        extension (str): File extension token (e.g. 'csv', 'xlsx', 'db').

    Returns:
        pandas.DataFrame: Loaded DataFrame (may be empty on read errors).

    Raises:
        ValueError: When an unsupported extension is passed.
    """
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
    """Read a CSV file into a DataFrame and normalize its column names.

    Args:
        file (io.BufferedIOBase): File-like object positioned at start.

    Returns:
        pandas.DataFrame: Parsed CSV content with stringified column names.

    Raises:
        ValueError: If the CSV is empty or cannot be parsed.
    """
    try:
        data = pd.read_csv(file)
        # Convert all column names to strings (handles numeric/mixed types)
        data.columns = data.columns.map(str)
        return data
    except UnicodeDecodeError:
        # pd.read_csv expects a UTF-8 by default
        file.seek(0)
        return pd.read_csv(file, encoding='latin-1')
    except pd.errors.EmptyDataError:
        # Specific error for empty CSV files
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError as e:
        # Parsing errors (malformed CSV, encoding issues, etc.)
        raise ValueError(f"Error parsing CSV: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error reading csv file {str(e)}")


def _upload_excel(file):
    """Read an Excel file into a DataFrame using ``openpyxl`` engine.

    Args:
        file (io.BufferedIOBase): File-like object representing the Excel
            workbook.

    Returns:
        pandas.DataFrame: Parsed sheet (first sheet) with string column names.

    Raises:
        ValueError: If there is any problem reading the Excel file.
    """
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
    """Load a SQLite database file into a DataFrame by reading the first table.

    The provided ``conn`` must be an in-memory sqlite3 connection. The
    function calls ``conn.deserialize`` using the uploaded file bytes to
    populate the in-memory database and then reads the first available table
    into a pandas DataFrame.

    Args:
        file (io.BufferedIOBase): Uploaded SQLite file-like object.
        conn (sqlite3.Connection): In-memory connection instance used to
            deserialize the database contents.

    Returns:
        pandas.DataFrame: Contents of the first table in the database.

    Raises:
        ValueError: If the database contains no tables or cannot be read.
    """
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

