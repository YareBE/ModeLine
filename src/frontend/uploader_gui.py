# First of all, we need to access the backend directory from the root
import streamlit as st

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from backend.uploader import (
    dataset_error_handler,
    load_sample_dataset,
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

    # Offer the bundled dataset, so the app can be tried without one to hand
    if uploaded_file is None:
        load_example()

    if st.session_state.file != uploaded_file:
        # An example dataset is loaded and nothing was uploaded: leave it be,
        # otherwise every rerun would reset it away
        if uploaded_file is None and st.session_state.get("sample_loaded"):
            return None

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
    st.session_state["sample_loaded"] = False


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


def load_example():
    """Offer the bundled example dataset as a one-click alternative to uploading.

    Without this, a first-time visitor meets an empty file uploader and has to
    find a dataset of their own before seeing the app do anything.

    Side effects:
        - Sets ``df`` and ``sample_loaded`` in ``st.session_state``.
    """
    st.caption("No dataset to hand?")
    if st.button("Load example dataset",
                 help="240 rows of synthetic house-price data, ready to model"):
        reset_session_state()
        try:
            st.session_state.df = load_sample_dataset()
        except Exception as e:
            st.error(f"Could not load the example dataset: {e}")
            return

        # No uploaded file backs this dataset, so mark it to survive reruns
        st.session_state.file = None
        st.session_state.sample_loaded = True
        st.success("✅ Example dataset loaded.")
