import streamlit as st
from data_manager import DataUploader
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from model_trainer import LRTrainer

MODELS_DIR = Path("models")

class Interface():
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            "dataframe": None,
            "selected_features": [],
            "selected_target": [],
            "processed_data": None,
            "model": None,
            "train_size": 100,
            "model_trained": False,
            "file_uploader_key": 0,
            "na_method" : None,
            "split_seed" : 1
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def load_file(self, data_file):
        """Load and process uploaded file - cached to avoid reloading"""
        try:
            selected_file = DataUploader(data_file)
            dataset = selected_file.error_handler()
            return dataset
        except Exception as err:
            return err

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_numeric_columns(df):
        """Cache numeric column detection"""
        return df.select_dtypes(include=['number']).columns.tolist()

    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_na_info(df):
        """Cache missing value detection"""
        cols_with_na = df.columns[df.isna().any()].tolist()
        return cols_with_na

    def apply_na_handling(self, _df, method, constant_value=None):
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

    def reset_downstream_selections(self, level):
        """Reset selections that depend on upstream choices"""
        if level <= 1:
            keys_to_reset = ["selected_features", "selected_target",
                           "processed_data", "model", "model_trained"]
            for key in keys_to_reset:
                if key == "selected_features" or key == "selected_target":
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None if key != "model_trained" else False
        elif level <= 2:
            keys_to_reset = ["processed_data", "description", "model", "model_trained", "na_method"]
            for key in keys_to_reset:
                st.session_state[key] = None if key != "model_trained" else False

    def load_models_section(self):
        """Model loading section without fragment"""
        if not MODELS_DIR.exists():
            return
            
        model_files = sorted([f for f in MODELS_DIR.glob("*.joblib")])
        model_names = [f.name for f in model_files]
        
        if model_names:
            st.write("or load an existing model")
            selected_model = st.selectbox("Select your model", model_names, key="model_selector")
            if st.button("Load model", key="load_model_btn"):
                model_path = MODELS_DIR / selected_model
                st.session_state.model = joblib.load(model_path)
                st.success(f"✅ '{selected_model}' correctly uploaded.")


    def style_dataframe(self, _df, features, target, *args, **kwargs):
        """Cache styled dataframe to avoid recalculation"""
        def highlight_columns(col, *args, **kwargs):
            if col.name in features:
                return ['background-color: #e3f2fd'] * len(col)
            elif col.name in target:
                return ['background-color: #ffb3b3'] * len(col)
            else:
                return [''] * len(col)
        
        return _df.style.apply(highlight_columns)

    def render_sidebar(self):
        """Render the complete sidebar with all controls"""
        with st.sidebar:
            st.title("Workflow")
            st.divider()
            
            # FILE UPLOAD
            st.subheader("1️⃣ Data Upload")
            uploaded_file = st.file_uploader(
                "Upload your dataset",
                type=["csv", "xls", "xlsx", "db", "sqlite"],
                help="Supported formats: CSV, Excel, SQLite",
                key=f"file_uploader_{st.session_state.file_uploader_key}"
            )
            
            if uploaded_file is not None and st.session_state.dataframe is None:
                with st.spinner("Loading data..."):
                    df = self.load_file(uploaded_file)
                    if not isinstance(df, pd.DataFrame):
                        st.error(str(df))
                        return
                    elif len(df) == 0:
                        st.error("ERROR: empty dataset")
                        return
                    else:
                        st.session_state.dataframe = df
                        self.reset_downstream_selections(1)
                        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
            
            if st.session_state.dataframe is not None:
                if st.button("🔄 Upload Different File", use_container_width=True,
                           help="Change dataset", key="reset_btn"):
                    # Clear all session state
                    for key in list(st.session_state.keys()):
                        if key != "file_uploader_key":
                            del st.session_state[key]
                    st.session_state.file_uploader_key += 1
                    self.initialize_session_state()
                    st.rerun()

            self.load_models_section()
            
            if st.session_state["model"] is None and st.session_state.dataframe is not None:
                df = st.session_state.dataframe
                available_columns = self.get_numeric_columns(df)

                st.divider()

                # DATASET INFO
                st.subheader("**DATASET INFO**")
                cols_with_na = self.get_na_info(df)

                col1, col2 = st.columns(2)
                col1.metric("Rows", len(df))
                col2.metric("Cols", len(df.columns))
                
                if cols_with_na:
                    st.caption(f"**NA Columns:** {', '.join(cols_with_na[:3])}{' ...' if len(cols_with_na) > 3 else ''}")

                if len(df) < 10:
                    st.warning("WARNING: As the dataset is too small, all of" \
                    " it will be used to train, resulting on an empty testing dataframe.")
                st.divider()

                # FEATURES SELECTION
                st.subheader("2️⃣ Features")
                selected_features = st.multiselect(
                    "Training Features (Numeric)",
                    options=[col for col in available_columns if col != st.session_state.selected_target],
                    default=st.session_state.selected_features,
                    help="Select training features",
                    key="features_multiselect"
                )
                
                if selected_features != st.session_state.selected_features:
                    st.session_state.selected_features = selected_features
                    self.reset_downstream_selections(2)
                
                st.divider()
                
                # TARGET SELECTION
                st.subheader("3️⃣ Target")
                target_options = [col for col in available_columns if col not in st.session_state.selected_features]
                
                if not target_options:
                    st.error("No variables left! Remove at least 1 feature")
                    return

                selected_target = st.multiselect(
                    "Target Variable (Numeric)",
                    options=[""] + target_options,
                    max_selections = 1,
                    help="Variable to predict",
                    key="target_selectbox"
                )
                
                if selected_target != st.session_state.selected_target:
                    st.session_state.selected_target = selected_target
                    self.reset_downstream_selections(2)
                
                if not selected_features or not selected_target:
                    st.info("Select features and target")
                    return
                
                st.divider()
                
                # MISSING VALUES HANDLING
                st.subheader("4️⃣ Handle NAs")
                selected_data = df[selected_features + selected_target]
                na_count = selected_data.isna().sum().sum()
                
                #The following conditional is to avoid showing the warning after
                #dropping NAs
                if (na_count > 0 and st.session_state.na_method != "Delete rows"\
                    and st.session_state.processed_data is None)\
                    or (na_count > 0 and st.session_state.processed_data is None):

                    st.warning(f"⚠️ {na_count} missing values")
                    
                    na_method = st.selectbox(
                        "Filling method",
                        options=["Select method...", "Delete rows", "Mean", "Median", "Constant"]
                    )
                    st.session_state.na_method = na_method
                    constant_value = None
                    if na_method == "Constant":
                        constant_value = st.text_input("Constant value", key="const_input")
                    
                    if st.button("Apply", type="primary", use_container_width=True, key="apply_na"):
                        if na_method == "Select method...":
                            st.error("Select a method")
                        elif na_method == "Constant" and not constant_value:
                            st.error("Enter a constant value")
                        else:
                            processed = self.apply_na_handling(selected_data, na_method, constant_value)
                            st.session_state.processed_data = processed
                            st.success("✅ NAs handled!")
                            st.rerun()
                else:
                    st.success("✅ No missing values")
                    if st.session_state.processed_data is None:
                        st.session_state.processed_data = selected_data
                
                if st.session_state.processed_data is None:
                    return
                
                st.divider()
                
                # TRAIN/TEST SPLIT
                train_size = 100 if len(df) < 10 else 80
                if len(df) >= 10:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        split_seed = st.number_input("Seed",
                                    help = "This seed will be used for a " \
                                    "random but repeteable split-generation",
                                    value = st.session_state.split_seed
                                )
                        st.session_state.split_seed = split_seed
                    with col2:
                        st.subheader("5️⃣ Split")
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
                
                total_rows = len(st.session_state.processed_data)
                train_rows = int(total_rows * train_size / 100)
                test_rows = total_rows - train_rows
                
                col1, col2 = st.columns(2)
                col1.metric("Train", f"{train_rows}")
                col2.metric("Test", f"{test_rows}")

    def display_dataframe(self):
        """Fragment for dataframe display to avoid full rerun on checkbox toggle"""
        df = st.session_state.dataframe.copy()
        if st.session_state.processed_data is not None:
            df[st.session_state.selected_features +\
                st.session_state.selected_target] = st.session_state.processed_data
        
        st.markdown("#### Dataset Preview")
        
        # Add controls for data display
        col1, col2 = st.columns([3, 1])
        with col1:
            available_rows = [[i, i + 100 if i+100 <= len(df) else len(df)] \
                            for i in range(0, len(df), 100)]
            rows_displayed = st.selectbox("Choose the range of rows to be" \
            " displayed", options = available_rows, index = 0 if 
            "rows_displayed" not in st.session_state else\
                    st.session_state.rows_displayed[0]//100)

            st.session_state.rows_displayed = rows_displayed
        
        # Display dataframe with limited rows for performance
        st.caption("🔵 Blue = Features | 🔴 Red = Target")
        
        # Only style and show limited rows
        display_df = df[st.session_state.rows_displayed[0]:\
                        st.session_state.rows_displayed[1]]
        styled_df = self.style_dataframe(
            _df = display_df,
            features = st.session_state.selected_features,
            target = st.session_state.selected_target,
            axis = 0
        )
        st.dataframe(styled_df, use_container_width=True, height=400)

    def visualize_results(self):
        """Display model results"""
        if "lr_trainer" in st.session_state and st.session_state.model_trained:
            
            st.divider()
            st.header("Model Results")
            formula = st.session_state.lr_trainer.get_formula()
            st.info(formula)
            
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Training Set")
                metrics_train = st.session_state.metrics['train']
                st.metric("R² Score", f"{metrics_train['r2']:.4f}")
                st.metric("MSE", f"{metrics_train['mse']:.4f}")
            with col2:
                if st.session_state.X_test is not None:
                        st.markdown("#### Test Set")
                        metrics_test = st.session_state.metrics['test']
                        st.metric("R² Score", f"{metrics_test['r2']:.4f}")
                        st.metric("MSE", f"{metrics_test['mse']:.4f}")
                else:
                    st.subheader("\tIMPORTANT")
                    st.warning("Remember, the performance is counting" \
                    " only the training set,\n so the results are not realistic")
                
            # Predictions plot
            st.divider()
            st.subheader("Predictions Visualization")
            fig = st.session_state.lr_trainer.plot_results(
            )
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

            st.subheader("Liked the performance? Save your model")
            st.text_input("Add an outline to be"
                " stored with your model",
                placeholder = 'Example: "Model for predicting body weight '
                ' based on height and age"', key = "model_description")
            st.warning("Adding a description is optional but recommended\n"
            " in order to avoid future confusions")

    def render_main_content(self):
        """Render the main content area"""
        st.title("ModeLine")
        st.header("Train and visualize linear regression models")
        st.divider()
        
        if st.session_state.dataframe is None:
            st.info("👈 Upload a dataset using the sidebar")
            st.markdown("""
            ### Getting Started
            1. Upload dataset (CSV, Excel, SQLite)
            2. Select features (numeric only)
            3. Choose target variable (numeric)
            4. Handle missing values if any
            5. Configure train/test split
            6. Train your model and visualize it!
            """)
            return

        self.display_dataframe()
        
        # MODEL TRAINING SECTION
        if st.session_state.processed_data is not None:
            st.success("✅ Data ready for training!")
            
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("TRAIN MODEL", type="primary", use_container_width=True, key="train_btn"):
                    with st.spinner("Training model..."):
                        # Prepare data
                        X = st.session_state.processed_data[st.session_state.selected_features]
                        y = st.session_state.processed_data[st.session_state.selected_target]
                        # Train model
                        lr_trainer = LRTrainer(X, y, float(st.session_state.train_size/100), st.session_state.split_seed)
                        X_train, X_test, y_train, y_test = lr_trainer.get_splitted_subsets()
                        model = lr_trainer.train_model()
                        metrics, y_train_pred, y_test_pred = lr_trainer.test_model()
                        
                        
                        # Store results
                        st.session_state.model = model
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.y_train_pred = y_train_pred
                        st.session_state.y_test_pred = y_test_pred
                        st.session_state.metrics = metrics
                        st.session_state.lr_trainer = lr_trainer
                        st.session_state.model_trained = True
                        
                        st.balloons()
            
            # Show results if model is trained
            if st.session_state.model_trained:
                self.visualize_results()

    def run(self):
        """Main application runner"""
        st.set_page_config(
            page_title="ModeLine",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.render_sidebar()
        self.render_main_content()


if __name__ == '__main__':
    app = Interface()
    app.run()