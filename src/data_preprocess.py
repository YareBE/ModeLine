import streamlit as st
from interface import reset_downstream_selections

@st.cache_data(show_spinner=False)
def get_numeric_columns(df):
    """Cache numeric column detection"""
    return df.select_dtypes(include=['number']).columns.tolist()

def get_na_info(df):
    """Cache missing value detection"""
    cols_with_na = df.columns[df.isna().any()].tolist()
    return cols_with_na

def apply_na_handling( _df, method, constant_value=None):
    """Cache NA handling operations"""
    df = _df.copy()
    if method == "Delete rows":
        return df.dropna()
    elif method == "Mean":
        return df.fillna(df.mean(numeric_only=True))
    elif method == "Median":
        return df.fillna(df.median(numeric_only=True))
    elif method == "Constant":
        if constant_value is not None:
            return df.fillna(constant_value)
    return df

def _dataset_info(df):
    available_columns = get_numeric_columns(df)
    cols_with_na = get_na_info(df)

    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Cols", len(df.columns))
    
    if cols_with_na:
        st.caption(f"**NA Columns:** {', '.join(cols_with_na[:3])}{' ...' if len(cols_with_na) > 3 else ''}")

    if len(df) < 10:
        st.warning("WARNING: As the dataset is too small, all of" \
        " it will be used to train, resulting on an empty testing dataframe.")
        st.session_state.trainset_only = True
    st.divider()

    return available_columns

def parameters_selection(df):
    available_columns = _dataset_info(df)
    st.subheader("2️⃣ Features")
    _features_selection(available_columns)
    st.divider()
    st.subheader("3️⃣ Target")
    _target_selection(available_columns)

    if not st.session_state.features or not st.session_state.target:
        st.info("Select features and target")
        return 

def _features_selection(available_columns):
    selected_features = st.multiselect(
        "Training Features (Numeric)",
        options=[col for col in available_columns if col != st.session_state.target],
        help="Select training features"
    )
    
    if selected_features != st.session_state.features:
        st.session_state.features = selected_features
        reset_downstream_selections(2)
    
 
    
def _target_selection(available_columns):
    target_options = [col for col in available_columns if col not in st.session_state.features]
    if not target_options:
        st.error("No variables left! Remove at least 1 feature")
        return
    
    selected_target = st.multiselect(
        "Target Variable (Numeric)",
        options=[""] + target_options,
        max_selections = 1,
        help="Variable to predict"
    )
    
    if selected_target != st.session_state.target:
        st.session_state.target = selected_target
        reset_downstream_selections(2)
    
    

def na_handler():
    
    selected_data = st.session_state.df[
        st.session_state.features +\
        st.session_state.target]
    na_count = selected_data.isna().sum().sum()
    
    #The following conditional is to avoid showing the warning after
    #dropping NAs
    if (na_count > 0 and st.session_state.processed_data is None):
        st.warning(f"⚠️ {na_count} missing values")
        
        na_method = st.selectbox(
            "Filling method",
            options=["Select method...", "Delete rows", "Mean", "Median", "Constant"]
        )
        st.session_state.na_method = na_method
        constant_value = None
        if na_method == "Constant":
            constant_value = st.text_input("Constant value")
        
        if st.button("Apply", type="primary", use_container_width=True):
            if na_method == "Select method...":
                st.error("Select a method")
            elif na_method == "Constant" and not constant_value:
                st.error("Enter a constant value")
            else:
                processed = apply_na_handling(selected_data, na_method, constant_value)
                if not processed.empty:
                    st.session_state.processed_data = processed
                st.write(st.session_state.processed_data)
        
                

    else:
        st.success("✅ No missing values")
        if st.session_state.processed_data is None:
            st.session_state.processed_data = selected_data
        return
    


def set_split():
    df = st.session_state.processed_data
    train_size = 100 if len(df) < 10 else 80
    if not st.session_state.trainset_only:
        col1, col2 = st.columns([1, 4])
        with col1:
            split_seed = st.number_input("Seed",
                        help = "This seed will be used for a " \
                        "random but repeteable split-generation",
                        value = st.session_state.seed
                    )
            st.session_state.split_seed = split_seed
        with col2:
            train_size = st.slider(
                "Training %",
                min_value=5,
                max_value=95,
                value=80,
                step=1,
                help="Train/test split percentage",
                key="train_slider"
            )
            st.session_state.train_size = train_size
    
    total_rows = len(df)
    train_rows = int(total_rows * train_size / 100)
    test_rows = total_rows - train_rows
    
    col1, col2 = st.columns(2)
    col1.metric("Train", f"{train_rows}")
    col2.metric("Test", f"{test_rows}")