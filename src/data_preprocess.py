import streamlit as st


def reset_downstream_selections(level):
    """Reset session state variables depending on upstream choices."""
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


@st.cache_data(show_spinner=False)
def get_numeric_columns(df):
    """Get numeric columns from DataFrame."""
    return df.select_dtypes(include=['number']).columns.tolist()


def get_na_info(df):
    """Get columns with missing values."""
    cols_with_na = df.columns[df.isna().any()].tolist()
    return cols_with_na


def apply_na_handling(df, method, constant_value=None):
    """Apply selected NA handling method to DataFrame."""
    df = df.copy()
    try:
        if method == "Delete rows":
            # Delete all the rows that contain at least a NA value
            return df.dropna()
        elif method == "Mean":
            # Fill with the mean, only in numeric columns
            return df.fillna(df.mean(numeric_only=True))
        elif method == "Median":
            # Fill with median, only in numeric columns
            return df.fillna(df.median(numeric_only=True))
        elif method == "Constant":
            if constant_value is not None:
                try:
                    # Transform the input value to float
                    value = float(constant_value)
                    return df.fillna(value)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Constant value must be numeric, got: {constant_value}"
                    )
        # If a valid method was not selected, return df without changes
        return df
    except Exception as e:
        raise RuntimeError(f"Error applying NA handling: {str(e)}")


def _dataset_info(df):
    """Display dataset info and return numeric columns."""
    available_columns = get_numeric_columns(df)
    cols_with_na = get_na_info(df)

    # Show basic metrics of the dataset
    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Cols", len(df.columns))
    
    # Show warning of missing values
    if cols_with_na:
        # Limit the list of the first 3 columns to not saturate the UI
        na_list = ', '.join(cols_with_na[:3])
        suffix = ' ...' if len(cols_with_na) > 3 else ''
        st.caption(f"**Columns with NA values:** {na_list}{suffix}")
    
    # Warn if the dataset is too small to do a split train/test
    if len(df) < 10:
        msg = ("WARNING: Dataset too small. Training set will contain "
               "all data, resulting in empty test set.")
        st.warning(msg)
        # Flag to use 100% of data in training
        st.session_state.trainset_only = True

    st.divider()
    return available_columns


def parameters_selection(df):
    """Handle feature and target selection from numeric columns."""
    available_columns = _dataset_info(df)

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
    """Handle feature selection UI."""
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
    """Handle target variable selection UI."""
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

def na_handler():
    """Handle missing values in selected features and target."""
    # Extract just the selected columns (features + target)
    selected_data = st.session_state.df[
        st.session_state.features + st.session_state.target
    ]
    # Count the total NA values in all the subset
    na_count = selected_data.isna().sum().sum()
    
    # Just show a warning if there are NAs and they have been not processed
    if na_count > 0 and st.session_state.processed_data is None:
        st.warning(f"⚠️ {na_count} missing values")
        
        # Selector of imputation method
        na_method = st.selectbox(
            "Filling method",
            options=["Select method...", "Delete rows", "Mean",
                     "Median", "Constant"]
        )
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
                processed = apply_na_handling(
                    selected_data, na_method, constant_value
                )
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
    """Configure train/test split ratio and display split info."""
    df = st.session_state.processed_data
    
    # If the dataset is not too small, allow split configuration
    if not st.session_state.trainset_only:
        col1, col2 = st.columns([1, 4])

        # Column 1: Input the seed for reproducibility
        with col1:
            st.number_input(
                "Seed",
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
    train_rows = int(total_rows * st.session_state.train_size / 100)
    test_rows = total_rows - train_rows
    
    # Show split metrics
    col1, col2 = st.columns(2)
    col1.metric("Train rows", f"{train_rows}")
    col2.metric("Test rows", f"{test_rows}")
