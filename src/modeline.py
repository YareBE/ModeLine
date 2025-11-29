"""ModeLine: Interactive Streamlit application for linear regression modeling.

This module contains the main Interface class that orchestrates the complete
workflow for loading data, preprocessing, feature selection, model training,
evaluation, and serialization. All methods include type hints and comprehensive
docstrings.
"""

import streamlit as st
from data_uploader import *
from data_preprocess import *
from model_trainer import *
from display_utils import *
from model_serializer import *
from typing import Optional, List, Dict, Any


class Interface:
    """Main Streamlit application interface for ModeLine.

    This class encapsulates the Streamlit UI layout and the workflow logic
    for loading data, preprocessing, training a linear regression model,
    visualizing results and serializing models. Persistent runtime state is
    stored in st.session_state.

    The interface coordinates data input → feature selection → preprocessing →
    train/test split → model training → evaluation → serialization.

    Attributes:
        None: Persistent values are stored in st.session_state rather
            than instance attributes so that Streamlit reruns preserve state.
    """

    def __init__(self) -> None:
        """Create an Interface and ensure Streamlit session state defaults.

        The constructor calls initialize_session_state which sets
        sensible defaults for required st.session_state keys if they are
        not already present. This enables multiple reruns of the app without
        losing user selections.
        """
        self.initialize_session_state()

    def initialize_session_state(self) -> None:
        """Populate missing keys in st.session_state with default values.

        This method only sets keys that are absent so any existing user
        selections remain intact across Streamlit reruns. Keys initialized
        include dataset reference, selected features/target, processed data,
        model object and other workflow controls.

        Returns:
            None
        """
        defaults = {
            "df": None,
            "features": [],
            "target": [],
            "model_trained": False,
            "processed_data": None,
            "na_method": None,
            "trainset_only": False,
            "model": None,
            "model_name": None,
            "file": None,
            "loaded_packet": None
        }

        # Only initialize missing keys to preserve existing state
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def reset_downstream_selections(self, level):
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


    def render_sidebar(self):
        """Render the Streamlit sidebar containing the workflow controls.

        The sidebar includes steps to upload data, pick features/target,
        handle missing values and configure the train/test split. Helpers
        such as **upload_file** and **parameters_selection** are invoked and
        will update **st.session_state** accordingly.

        Returns:
            None
        """
        # 1. Data upload
        with st.sidebar:
            st.title("Workflow")
            st.divider()
            st.subheader("1️⃣ Data Upload")
            self.upload_file()
            df = st.session_state.df

            # Only show subsequent steps if data is loaded
            if df is not None:
                st.divider()

                # 2. Feature and target selection
                st.subheader("2️⃣ Dataset Info")
                self.parameters_selection(df)
                st.divider()

                # 3. Missing values handling
                # Only visible after features and target are selected
                if st.session_state.features and st.session_state.target:
                    st.subheader("4️⃣ Handle NAs")
                    self.na_handling_selection()

                # 4. Train/test split configuration
                # Only visible after NA handling is complete
                if st.session_state.processed_data is not None:
                    st.divider()
                    st.subheader("5️⃣ Split")
                    self.set_split()


    def render_main_content(self):
        """Render the main application area (data preview, training, results).

        Behavior depends on the current **st.session_state**: if a model
        packet has been loaded, the packet UI and prediction UI are shown.
        Otherwise the function renders the data preview and, when the data
        are ready, exposes the model training button and results panels.

        Returns:
            None
        """
        st.title("ModeLine")
        st.header("Train, visualize and predict with linear regression models")
        st.divider()

        # Display loaded model instead of training workflow
        if st.session_state.loaded_packet is not None:
            display_uploaded_model()
            self.predict()
            return

        # Display getting started guide
        if st.session_state.df is None:
            st.info("👈 Upload a dataset/model using the sidebar")
            st.markdown("""
                ### Getting Started
                1. Upload dataset (CSV, Excel, SQLite) or model (joblib)
                2. Select features (numeric)
                3. Choose target variable (numeric)
                4. Handle missing values if any
                5. Configure train/test split and seed
                6. Train your model and visualize it
                7. Make predictions with your trained or uploaded model
                8. Save it in your device (if desired)
            """)
            return
        
        # Display data preview
        display_dataframe()

        # Model training section
        if (st.session_state.processed_data is not None and
                st.session_state.model is None):
            st.success("✅ Data ready for training!")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "TRAIN MODEL", type="primary",
                    use_container_width=True
                ):
                    # Trigger model training
                    self.train_model()

        # Display results after model is trained
        # Only show if both model and data exist
        if st.session_state.model is not None and st.session_state.df is not None:
            st.divider()
            st.header("Model Results")

            # Display metrics, formula, and performance statistics
            visualize_results()

            st.divider()
            self.predict()

            st.divider()
            st.subheader("Predictions Visualization")

            # Display scatter plots comparing predictions vs actual values
            plot_results()

        # Store model section 
        # Always visible if model exists, independent of reruns
        if st.session_state.model is not None:
            self.store_model()

    def upload_file(self):
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
            else:
                # Handle model files (Joblib)
                with st.spinner("Loading data..."):
                    try:
                        # Load serialized model packet
                        st.session_state.loaded_packet = upload_joblib(uploaded_file)

                    except InvalidJoblibPacket as e:
                        st.error(e)

                    else:             
                        # Store model name without extension
                        st.session_state.model_name = (
                            uploaded_file.name.replace('.joblib', '')
                        )
                        st.success("✅ Model correctly loaded.")


    def parameters_selection(self, df):
        """Render UI controls in Streamlit to select features and target.

        This function relies on Streamlit session state to store the selected
        **features** and **target**. It shows dataset info, then two subsections to
        pick training features and the target variable (numeric-only).

        Args:
            df (pandas.DataFrame): Loaded dataset used to determine available
                numeric columns and display dataset metrics.
        """
        available_columns = display_dataset_info(df)

        # Section to select the features
        st.subheader("2️⃣ Features")
        self._features_selection(available_columns)
        st.divider()

        # Section to select the target
        st.subheader("3️⃣ Target")
        self._target_selection(available_columns)

        # Validation: both selections must be complete
        if not st.session_state.features or not st.session_state.target:
            st.info("Select features and target")
            return


    def _features_selection(self, available_columns):
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
            on_change=lambda: self.reset_downstream_selections(2),
            key="features"
        )


    def _target_selection(self, available_columns):
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
            on_change=lambda: self.reset_downstream_selections(2),
            key="target"
        )



    def na_handling_selection(self):
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


    def set_split(self):
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
                    "Seed",
                    help="Seed for reproducible split",
                    key="seed",
                    value=1,
                    # Reset the model when split changes
                    on_change=lambda: self.reset_downstream_selections(3)
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
                    on_change=lambda: self.reset_downstream_selections(3)
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

    def predict(self):
        """Render a prediction UI and compute a single prediction."""
        st.subheader("Predict with ModeLine")
        st.markdown("##### Enter the input values for your prediction")

        features = (
            st.session_state.features
            or st.session_state.loaded_packet.get("features")
        )
        target = (
            st.session_state.target
            or st.session_state.loaded_packet.get("target")
        )

        if not features or not target:
            st.error("Features or target not found in session state")
            return  
        
        model = st.session_state.get("model") or st.session_state.loaded_packet.get("model")
        if model is None:
            st.error("No model found. Train or load a model first")
            return
        
        columns = {}
        for i in range(len(features)):
            if i % 4 == 0:
                columns[str(i//4)] = st.columns(np.ones(4))
        inputs = np.ones(len(features))

        for name in features:
            index = features.index(name)
            with columns[str(index//4)][index % 4]:
                inputs[index] = st.number_input(name, value=1.0, step=0.1)

        if st.button("PREDICT", type="primary"):
            try:
                prediction = modeline_prediction(model, inputs)
                st.markdown(f"#### Predicted {target[0]}: {prediction:,.3f}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    def train_model(self):
        """Train a Linear Regression model using the current processed data.

        This method extracts **X** and **y** from
        **st.session_state.processed_data** using the user-selected column
        names, configures the train/test split based on session settings and
        uses **LRTrainer** to perform splitting, fitting and evaluation.

        Side effects:
            Writes trained **model**, evaluation **metrics**, predictions
            (**y_train_pred**, **y_test_pred**), readable **formula** and
            train/test subsets into **st.session_state**.

        Returns:
            None
        """
        with st.spinner("Training model..."):
            # Extract features and target from processed data
            X = st.session_state.processed_data[st.session_state.features]
            y = st.session_state.processed_data[st.session_state.target]
            
            # Train model
            if not st.session_state.trainset_only:
                # Normal split: use user-configured train percentage and seed
                train_ratio = st.session_state.train_size / 100
                seed = st.session_state.seed
            else:
                # Small dataset: use all data for training
                train_ratio = 1
                seed = 0

            # Store train/test subsets in session state
            (st.session_state.X_train, st.session_state.X_test,
             st.session_state.y_train, st.session_state.y_test) = (
                 split_dataset(X, y, train_ratio, seed)
                
            )

            # Train model and store in session state
            st.session_state.model = train_linear_regression(st.session_state.X_train,
                                            st.session_state.y_train)

            # Evaluate model and store metrics and predictions
            (st.session_state.y_train_pred, st.session_state.y_test_pred, \
            st.session_state.metrics, ) = evaluate_model(st.session_state.model, \
                st.session_state.X_train, st.session_state.y_train, \
                st.session_state.X_test, st.session_state.y_test)

            # Generate and store a readable formula
            st.session_state.formula = generate_formula(st.session_state.model, \
                    st.session_state.features, st.session_state.target[0])

            st.balloons()
    
    def store_model(self):
        """Render UI to download the trained model and its metadata as joblib.

        The function displays input controls for an optional description and a
        filename. It packages the model object and metadata stored in
        ``st.session_state`` into a single dictionary and serializes it to a
        Joblib binary which is available via a Streamlit download button.

        Side effects:
                    - Reads ``st.session_state`` keys such as ``model``, ``features``,
                        ``target``, ``formula`` and ``metrics``.
            - When the user clicks the download button, returns the serialized
            bytes as a downloadable file.

        Returns:
            None

        Raises:
            Exception: Any error during serialization or download preparation
                is caught and displayed to the user via **st.error**.
        """
        st.subheader("📦 Download Your Model")

        # Create a two column display for input fields
        col1, col2 = st.columns([2, 1])
        
        # Column 1: Description text area
        with col1:
            # Optional description to help users remember purpose and context
            st.text_area(
                "Description (optional but recommended)",
                placeholder='Example: "Model for predicting body weight"',
                height=80, 
                key = "description"
            )
        
        # Column 2: Filename input
        with col2:
            # Filename input with default value
            filename = st.text_input(
                "File name",
                value="exported_model"
            )

        # Validate filename and prepare download
        if not filename or filename.strip() == '':
            st.error("❌ File name cannot be empty!")
            return

        try:    
            buffer = packet_creation(st.session_state.model, st.session_state.description, st.session_state.features, 
                        st.session_state.target, st.session_state.formula, st.session_state.metrics)
            # Create download button with serialized data
            if st.download_button(
                label="⬇️ DOWNLOAD MODEL",
                data=buffer.getvalue(),           # Get bytes from buffer
                file_name=f"{filename}.joblib",   # Append .joblib extension
                mime="application/octet-stream",  # Binary file type
                type="primary",                 
                use_container_width=True        
            ):
                st.success(f"✅ Model {filename} downloaded correctly!")

        except Exception as e:
            # Catch any errors during serialization or download preparation
            st.error(f"❌ Error: {str(e)}")

    def run(self):
        """Configure Streamlit page and render the sidebar and main content.

        This is the entry point for running the Streamlit app UI. It sets
        the page configuration (title, layout) and delegates rendering to
        **render_sidebar** and **render_main_content**.

        Returns:
            None
        """
        st.set_page_config(
            page_title="ModeLine",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Render both main sections of the application
        self.render_sidebar()
        self.render_main_content()


if __name__ == '__main__':
    # Create and run the interface
    app = Interface()
    app.run()