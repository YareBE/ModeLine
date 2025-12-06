# First of all, we need to access the backend directory from the root
from display_utils import display_dataset_info
import streamlit as st

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from backend.data_preprocess import (
    apply_na_handling
)


def reset_downstream_selections(level):
    """Reset downstream session state variables when upstream choices change.

    This function resets Streamlit ``st.session_state`` keys that depend on
    earlier steps of the workflow. The granularity of the reset is
    controlled by the ``level`` parameter: lower levels perform broader
    resets.

    Args:
        level (int): Reset granularity. Typical values:
            - 1: Reset dataset, features and target selections.
            - 2: Additionally reset processed data and NA method.
            - 3: Additionally reset description and model.
    """
    if level <= 1:
        # Completely reset the session_state when the dataset changes
        keys_to_reset = ["features", "target", "df"]
        for key in keys_to_reset:
            if key in ("features", "target"):
                # Empty list for multiple selections
                st.session_state[key] = []
    if level <= 2:
        keys_to_reset = ["processed_data", "na_method"]
        for key in keys_to_reset:
            st.session_state[key] = None
    if level <= 3:
        keys_to_reset = ["description", "model"]
        for key in keys_to_reset:
            st.session_state[key] = None


def parameters_selection(df):
    """Render UI controls in Streamlit to select features and target.

    This function relies on Streamlit session state to store the selected
    **features** and **target**. It shows dataset info, then two subsections to
    pick training features and the target variable (numeric-only).

    Args:
        df (pandas.DataFrame): Loaded dataset used to determine available
            numeric columns and display dataset metrics.
    """
    available_columns = display_dataset_info(df)

    if len(available_columns) <= 0:
        st.error("⚠️ The uploaded dataset doesn't contain numeric "
                 "columns. Please load another file.")
        st.stop()

    # Section to select the features
    st.subheader("2️⃣ Features")
    _features_selection(available_columns)
    st.divider()

    # Section to select the target
    st.subheader("3️⃣ Target")
    _target_selection(available_columns)

    # Validation: both selections must be complete
    if not st.session_state.features or not st.session_state.target:
        st.info("Select features and target")
        return


def _features_selection(available_columns):
    """Render the feature selection control in the sidebar.

    Args:
        available_columns (list): List of numeric column names available for
            selection. The currently selected target (if any) is excluded from
            the options to avoid choosing the same column as both feature and
            target.
    """
    st.multiselect(
        "Training Features (Numeric)",
        # Exclude the actual target from the options
        options=[col for col in available_columns
                 if col != st.session_state.target],
        help="Select training features",
        # Reset processing downstream when features are changed
        on_change=lambda: reset_downstream_selections(2),
        key="features"
    )


def _target_selection(available_columns):
    """Render the target variable selection control in the sidebar.

    Args:
        available_columns (list): List of numeric column names available for
            selection. Already-selected features are excluded from the target
            options.
    """
    # Exclude the selected features
    target_options = [col for col in available_columns
                      if col not in st.session_state.features]

    # Validation: There must be at least one column for target
    if not target_options:
        st.error("No variables left! Remove at least 1 feature")
        return

    st.multiselect(
        "Target Variable (Numeric)",
        # Include an empty option to allow deselection
        options=[""] + target_options,
        max_selections=1,  # Just a variable target can be selected
        help="Variable to predict",
        # Reset processing downstream when the target changes
        on_change=lambda: reset_downstream_selections(2),
        key="target"
    )


def na_handling_selection():
    """Provide UI to inspect and handle missing values for selected subset.

    The function extracts the current selection of features and target from
    ``st.session_state`` and reports the total number of missing values. If
    missing values are present and have not been processed yet, the user is
    offered several strategies (delete rows, mean, median, constant) to
    handle them. The processed result is saved to
    ``st.session_state.processed_data``.
    """
    # Extract just the selected columns (features + target)
    selected_data = st.session_state.df[
        st.session_state.features + st.session_state.target
    ]
    # Count the total NA values in all the subset
    na_count = selected_data.isna().sum().sum()

    # Just show a warning if there are NAs and they have been not processed
    if na_count > 0 and st.session_state.processed_data is None:
        st.warning(f"⚠️ {na_count} missing values")

        # Calculate how many rows would be deleted
        rows_with_na = selected_data.isna().any(axis=1).sum()
        total_rows = len(selected_data)

        options = ["Select method...", "Mean", "Median", "Constant"]
        if rows_with_na < total_rows:
            # Some rows are complete, "Delete rows" is safe
            options.insert(1, "Delete rows")
        else:
            # All rows have NA, can't use "Delete rows"
            st.warning(
                "All rows contain NA values. 'Delete rows' option is disabled.")
        na_method = st.selectbox("Filling method", options=options)

        st.session_state.na_method = na_method
        constant_value = None

        # If the Constant method is selected, ask for the value
        if na_method == "Constant":
            constant_value = st.text_input("Constant value")

        # Button to apply the selected method
        if st.button("Apply", type="primary", use_container_width=True):
            # Validation: there must be a selected method
            if na_method == "Select method...":
                st.error("Select a method")
            # Validation: if the method is Constant, there must a value
            elif na_method == "Constant" and not constant_value:
                st.error("Enter a constant value")
            else:
                # Apply the imputation method
                try:
                    processed = apply_na_handling(
                        selected_data, na_method, constant_value
                    )
                except Exception as e:
                    st.error(f"error: {str(e)}")
                    return
                # Just save if there are still data after processing
                if not processed.empty:
                    st.session_state.processed_data = processed
                # Rerun the app after updating the UI
                st.rerun()
    else:
        # There are no more NA or they have been processed
        st.success("✅ No missing values")
        # If processed data had not been saved, save actual
        if st.session_state.processed_data is None:
            st.session_state.processed_data = selected_data
        return


def set_split():
    """Render controls for train/test split and display split metrics.

    The function uses **st.session_state.processed_data** to compute the
    number of rows assigned to training and testing according to the
    configured **st.session_state.train_size** and optionally the seed. For
    very small datasets a 100% training split is enforced and flagged via
    **st.session_state.trainset_only**.
    """
    df = st.session_state.processed_data

    # If the dataset is not too small, allow split configuration
    if not st.session_state.trainset_only:
        col1, col2 = st.columns([1, 4])

        # Column 1: Input the seed for reproducibility
        with col1:
            st.number_input(
                label="Seed",
                min_value=0,
                help="Seed for reproducible split",
                key="seed",
                value=1,
                # Reset the model when split changes
                on_change=lambda: reset_downstream_selections(3)
            )

        # Column 2: Slider to select percentage of train
        with col2:
            st.slider(
                "Training %",
                min_value=5,
                max_value=95,
                value=80,
                step=1,
                help="Train/test split percentage",
                key="train_size",
                # Reset the model when split changes
                on_change=lambda: reset_downstream_selections(3)
            )
    else:
        # If the dataset is small use 100% of it for training
        st.session_state.train_size = 100

    # Calculate the number of rows for each set
    total_rows = len(df)
    train_rows = max(1,
                     int(total_rows * st.session_state.train_size / 100))
    test_rows = total_rows - train_rows

    # Show split metrics
    col1, col2 = st.columns(2)
    col1.metric("Train rows", f"{train_rows}")
    col2.metric("Test rows", f"{test_rows}")
