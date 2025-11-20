import streamlit as st
from data_uploader import upload_file
from data_preprocess import (
    parameters_selection, na_handler, set_split, reset_downstream_selections
)
from model_trainer import LRTrainer, predict
from display_utils import ( 
            display_dataframe, visualize_results, plot_results
)
from model_serializer import (
            store_model, upload_model
)


class Interface:
    """Main Streamlit application interface for ModeLine."""

    def __init__(self):
        """Initialize the interface and session state."""
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all required session state variables."""
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
            "loaded_packet": None,
            "prediction_result": None,
            "prediction_inputs": {}
        }

        # Only initialize missing keys to preserve existing state
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render_sidebar(self):
        """Render the sidebar with workflow controls."""
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
        """Render the main content area."""
        st.title("ModeLine")
        st.header("Train and visualize linear regression models")
        st.divider()

        # Display loaded model instead of training workflow
        if st.session_state.loaded_packet is not None:
            upload_model()
            predict()
            return

        # Display getting started guide
        if st.session_state.df is None:
            st.info("👈 Upload a dataset using the sidebar")
            st.markdown("""
                ### Getting Started
                1. Upload dataset (CSV, Excel, SQLite)
                2. Select features (numeric only)
                3. Choose target variable (numeric)
                4. Handle missing values if any
                5. Configure train/test split
                6. Train your model and visualize it!
                7. Make predictions with your trained model
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
        """Train the linear regression model with current data."""
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

            # Initialize trainer with data and split configuration
            trainer = LRTrainer(X, y, train_ratio, seed)

            # Store train/test subsets in session state
            (st.session_state.X_train, st.session_state.X_test,
             st.session_state.y_train, st.session_state.y_test) = (
                trainer.get_splitted_subsets()
            )

            # Train model and store in session state
            st.session_state.model = trainer.train_model()

            # Evaluate model and store metrics and predictions
            (st.session_state.metrics, st.session_state.y_train_pred,
             st.session_state.y_test_pred) = trainer.test_model()

            # Generate and store a readable formula
            st.session_state.formula = trainer.get_formula()

            # Reset prediction state when new model is trained
            st.session_state.prediction_result = None
            st.session_state.prediction_inputs = {}

            st.balloons()

    def run(self):
        """Run the main Streamlit application."""
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
