import streamlit as st
from data_uploader import upload_file
from data_preprocess import (
    parameters_selection, na_handler, set_split
)
from model_trainer import *
from display_utils import ( 
            display_dataframe, visualize_results, plot_results, display_uploaded_model
)
from model_serializer import (
            store_model
)


class Interface:
    """Main Streamlit application interface for ModeLine.

    This class encapsulates the Streamlit UI layout and the workflow logic
    for loading data, preprocessing, training a linear regression model,
    visualizing results and serializing models. Persistent runtime state is
    stored in **st.session_state**.

    Attributes:
        None: Persistent values are stored in **st.session_state** rather
            than instance attributes so that Streamlit reruns preserve state.
    """

    def __init__(self):
        """Create an Interface and ensure Streamlit session state defaults.

        The constructor calls **initialize_session_state** which sets
        sensible defaults for required **st.session_state** keys if they are
        not already present. This enables multiple reruns of the app without
        losing user selections.
        """
        self.initialize_session_state()

    def initialize_session_state(self):
        """Populate missing keys in **st.session_state** with default values.

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
            upload_file()
            df = st.session_state.df

            # Only show subsequent steps if data is loaded
            if df is not None:
                st.divider()

                # 2. Feature and target selection
                st.subheader("2️⃣ Dataset Info")
                parameters_selection(df)
                st.divider()

                # 3. Missing values handling
                # Only visible after features and target are selected
                if st.session_state.features and st.session_state.target:
                    st.subheader("4️⃣ Handle NAs")
                    na_handler()

                # 4. Train/test split configuration
                # Only visible after NA handling is complete
                if st.session_state.processed_data is not None:
                    st.divider()
                    st.subheader("5️⃣ Split")
                    set_split()


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
            predict()
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
                    self._train_model()

        # Display results after model is trained
        # Only show if both model and data exist
        if st.session_state.model is not None and st.session_state.df is not None:
            st.divider()
            st.header("Model Results")

            # Display metrics, formula, and performance statistics
            visualize_results()

            st.divider()
            predict()

            st.divider()
            st.subheader("Predictions Visualization")

            # Display scatter plots comparing predictions vs actual values
            plot_results()

        # Store model section 
        # Always visible if model exists, independent of reruns
        if st.session_state.model is not None:
            store_model()

    def _train_model(self):
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