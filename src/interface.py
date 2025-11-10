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
            keys_to_reset = ["features", "target", "model"]
            for key in keys_to_reset:
                if key == "features" or key == "target":
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        elif level <= 2:
            keys_to_reset = ["processed_data","description", "model", "na_method"]
            for key in keys_to_reset:
                st.session_state[key] = None
class Interface:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            "df": None,
            "features": [],
            "target": [],
            "model_trained": False,
            "processed_data" : None,
            "na_method" : None,
            "trainset_only" : False,
            "model" : None
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
        else:
            display_dataframe()
        
        # MODEL TRAINING SECTION
        if st.session_state.processed_data is not None and st.session_state.model is None:
            st.success("✅ Data ready for training!")
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("TRAIN MODEL", type="primary", use_container_width=True, key="train_btn"):
                    with st.spinner("Training model..."):
                        # Prepare data
                        X = st.session_state.processed_data[st.session_state.features]
                        y = st.session_state.processed_data[st.session_state.target]
                        # Train model
                        lrt = LRTrainer(X, y, float(st.session_state.train_size/100), st.session_state.seed)
                        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = lrt.get_splitted_subsets()
                        st.session_state.model = lrt.train_model()
                        st.session_state.metrics, st.session_state.y_train_pred, st.session_state.y_test_pred = lrt.test_model()
                        st.session_state.formula = lrt.get_formula()
                        st.balloons()
            
        if st.session_state.model is not None:
            st.divider()
            st.header("Model Results")
            visualize_results()

            st.divider()
            st.subheader("Predictions Visualization")
            plot_results()

            st.divider()
            if st.session_state.model is not None and st.session_state.df is not None:
                store_model()

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