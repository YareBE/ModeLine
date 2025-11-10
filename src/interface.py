import streamlit as st
from data_uploader import upload_file
import pandas as pd
import numpy as np
from data_preprocess import *
from model_trainer import LRTrainer
from display_utils import *
from model_serializer import *


def reset_downstream_selections(level):
        """Reset selections that depend on upstream choices"""
        if level <= 1:
            keys_to_reset = ["features", "target", "model_trained"]
            for key in keys_to_reset:
                if key == "features" or key == "target":
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None if key != "model_trained" else False
        elif level <= 2:
            keys_to_reset = ["processed_data","description", "model_trained", "na_method"]
            for key in keys_to_reset:
                st.session_state[key] = None if key != "model_trained" else False
class Interface:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            "df": None,
            "features": [],
            "target": [],
            "train_size": 100,
            "model_trained": False,
            "processed_data" : None,
            "na_method" : None,
            "seed" : 1,
            "trainset_only" : False,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value


    def render_sidebar(self):
        """Render the complete sidebar with all controls"""
        with st.sidebar:
            st.title("Workflow")
            st.divider()
            st.subheader("1️⃣ Data Upload")
            upload_file()
            df = st.session_state.df

            if df is not None:
                st.divider()
                st.subheader("**DATASET INFO**")
                parameters_selection(df)
                st.divider()
                
                # MISSING VALUES HANDLING
                if st.session_state.features and st.session_state.target:
                    st.subheader("4️⃣ Handle NAs")
                    na_handler()
               

                if st.session_state.processed_data is not None:
                    st.divider()
                    st.subheader("5️⃣ Split")
                    set_split()
                    


    def render_main_content(self):
        """Render the main content area"""
        st.title("ModeLine")
        st.header("Train and visualize linear regression models")
        st.divider()
        
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
            """)
            return

        display_dataframe()
        
        # MODEL TRAINING SECTION
        if st.session_state.processed_data is not None and st.session_state.model is not None:
            st.success("✅ Data ready for training!")
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("TRAIN MODEL", type="primary", use_container_width=True, key="train_btn"):
                    with st.spinner("Training model..."):
                        features = st.session_state.features
                        target = st.session_state.target
                        # Prepare data
                        X = st.session_state.processed_data[features]
                        y = st.session_state.processed_data[target]
                        # Train model
                        lrt = LRTrainer(X, y, float(st.session_state.train_size/100), st.session_state.seed)
                        X_train, X_test, y_train, y_test = lrt.get_splitted_subsets()
                        model = lrt.train_model()
                        metrics, y_train_pred, y_test_pred = lrt.test_model()
                        formula = lrt.get_formula()
                        st.session_state.model_trained = True
                        st.balloons()
            
            if st.session_state.model_trained:
                st.divider()
                st.header("Model Results")
                visualize_results(formula, metrics)

                st.divider()
                st.subheader("Predictions Visualization")
                plot_results(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)

                st.divider()
                if st.session_state.model_trained:
                    store_model(model, features, target, formula, metrics)

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