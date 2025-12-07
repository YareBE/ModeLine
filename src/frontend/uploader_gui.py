# First of all, we need to access the backend directory from the root
import streamlit as st

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from backend.uploader import (
    dataset_error_handler,
    upload_joblib
)  # noqa: E402


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
        reset_session_state()

        # Store uploaded file reference
        st.session_state.file = uploaded_file
        if uploaded_file is None:
            return None

        # Extract file extension for format detection
        extension = uploaded_file.name.split('.')[-1].lower()
        if extension != "joblib":
            handle_data_file(uploaded_file, extension)
        else:
            # Handle model files (Joblib)
            handle_model_file(uploaded_file)


def reset_session_state():
    """Reset relevant session_state keys when the file changes"""
    for key in st.session_state:
        if key == "features" or key == "target":
            # Reset selection lists to empty
            st.session_state[key] = []
        elif key in ["processed_data", "description", "model", "na_method",
                     "df", "loaded_packet"]:
            # Reset objects to None
            st.session_state[key] = None
    st.session_state["trainset_only"] = False


def handle_data_file(uploaded_file, extension):
    """Handle loading of data (CSV, Excel, SQLite)"""
    try:
        # Handle datafile (CSV, Excel and SQLite)
        df = dataset_error_handler(uploaded_file, extension)
    except Exception as e:
        st.error(f"{e}. Try a new file.")
    else:
        with st.spinner("Loading data..."):
            # Store DataFrame in session state
            st.session_state.df = df
            st.success("✅ Dataset correctly loaded.")


def handle_model_file(uploaded_file):
    """Handle loading of model file (Joblib)"""
    with st.spinner("Loading data..."):
        try:
            # Load serialized model packet
            st.session_state.loaded_packet = upload_joblib(uploaded_file)
        except Exception as e:
            st.error(f"{e}. Try a new file.")
        else:
            # Store model name without extension
            st.session_state.model_name = (
                uploaded_file.name.replace('.joblib', '')
            )
            st.success("✅ Model correctly loaded.")
