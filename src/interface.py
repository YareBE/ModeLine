import streamlit as st
from data_manager import DataUploader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from model_trainer import LRTrainer
import plotly.express as px
import plotly.graph_objects as go

class Interface():
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        if "dataframe" not in st.session_state:
            st.session_state.dataframe = None
        if "selected_features" not in st.session_state:
            st.session_state.selected_features = []
        if "selected_target" not in st.session_state:
            st.session_state.selected_target = None
        if "na_method" not in st.session_state:
            st.session_state.na_method = None
        if "train_size" not in st.session_state:
            st.session_state.train_size = 80
        if "processed_data" not in st.session_state:
            st.session_state.processed_data = None
        if "model" not in st.session_state:
            st.session_state.model = None
        if "model_trained" not in st.session_state:
            st.session_state.model_trained = False

    @staticmethod
    @st.cache_data
    def load_file(data_file):
        """Load and process uploaded file"""
        try:
            selected_file = DataUploader(data_file)
            dataset = selected_file.error_handler()
            return dataset
        except Exception as err:
            return err

    def reset_downstream_selections(self, level):
        """Reset selections that depend on upstream choices"""
        if level <= 1:
            st.session_state.selected_features = []
            st.session_state.selected_target = None
            st.session_state.na_method = None
            st.session_state.processed_data = None
            st.session_state.model = None
            st.session_state.model_trained = False
        elif level <= 2:
            st.session_state.na_method = None
            st.session_state.processed_data = None
            st.session_state.model = None
            st.session_state.model_trained = False

    def style_dataframe(self, df):
        """Apply color styling to dataframe based on selected features and target"""
        if df is None:
            return None
        
        def highlight_columns(col):
            if col.name in st.session_state.selected_features:
                return ['background-color: #e3f2fd'] * len(col)
            elif col.name == st.session_state.selected_target:
                return ['background-color: #ffebee'] * len(col)
            else:
                return [''] * len(col)
        
        return df.style.apply(highlight_columns)

    def handle_missing_values(self, df, method, constant_value=None):
        """Handle missing values according to selected method"""
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

    def train_model(self, X_train, y_train):
        """Train the linear regression model"""
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Evaluate model performance"""
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'train': {
                'r2': r2_score(y_train, y_train_pred),
                'mse': mean_squared_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred)
            },
            'test': {
                'r2': r2_score(y_test, y_test_pred),
                'mse': mean_squared_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae': mean_absolute_error(y_test, y_test_pred)
            }
        }
        
        return metrics, y_train_pred, y_test_pred

    def plot_predictions(self, y_train, y_train_pred, y_test, y_test_pred):
        """Create visualization of predictions vs actual values"""
        # Combine data
        train_df = pd.DataFrame({
            'Actual': y_train,
            'Predicted': y_train_pred,
            'Set': 'Train'
        })
        
        test_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_test_pred,
            'Set': 'Test'
        })
        
        combined_df = pd.concat([train_df, test_df])
        
        # Create scatter plot
        fig = px.scatter(
            combined_df,
            x='Actual',
            y='Predicted',
            color='Set',
            title='Predictions vs Actual Values',
            labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
            color_discrete_map={'Train': '#1f77b4', 'Test': '#ff7f0e'}
        )
        
        # Add perfect prediction line
        min_val = min(combined_df['Actual'].min(), combined_df['Predicted'].min())
        max_val = max(combined_df['Actual'].max(), combined_df['Predicted'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        return fig

    def render_sidebar(self):
        """Render the complete sidebar with all controls"""
        st.sidebar.title("Workflow")
        st.sidebar.divider()
        
        # File upload
        st.sidebar.subheader("1️⃣ Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your dataset",
            type=["csv", "xls", "xlsx", "db", "sqlite"],
            help="Supported formats: CSV, Excel, SQLite"
        )
        
        if uploaded_file is not None:
            if st.session_state.dataframe is None:
                with st.spinner("Loading data..."):
                    df = self.load_file(uploaded_file)
                    if type(df) != pd.DataFrame:
                        st.sidebar.error(df)
                        return None
                    elif len(df) == 0:
                        st.sidebar.error("ERROR: empty dataset")
                        return 
                    else:
                        st.session_state.dataframe = df
                        self.reset_downstream_selections(1)
                        st.sidebar.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        if st.session_state.dataframe is not None:
            if st.sidebar.button("🔄 Upload Different File", use_container_width=True,
                                 help="To change the dataset, browse and drop a new one and then click this button"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        if st.session_state.dataframe is None:
            return
        
        df = st.session_state.dataframe
        available_columns = df.select_dtypes(include=['number']).columns.tolist()

        st.sidebar.divider()
        
        # FEATURES SELECTION
        st.sidebar.subheader("2️⃣ Feature Selection")
        selected_features = st.sidebar.multiselect(
            "Select Training Features (Numeric only!)",
            options=[col for col in available_columns if col != st.session_state.selected_target],
            default=[],
            help="Choose one or more features for training",
            key="features_multiselect"
        )
        
        if selected_features != st.session_state.selected_features:
            st.session_state.selected_features = selected_features
            self.reset_downstream_selections(2)
        
        st.sidebar.divider()
        
        # TARGET SELECTION
        st.sidebar.subheader("3️⃣ Target Selection")
        target_options = [col for col in available_columns if col not in st.session_state.selected_features]
        
        if len(target_options) == 0:
            st.sidebar.error("ERROR: There is no variables left! Reject at least 1 feature")
            return 

        selected_target = st.sidebar.selectbox(
            "Select Target Variable (Numeric only!)",
            options=[""] + target_options,
            index=0 if st.session_state.selected_target is None else target_options.index(st.session_state.selected_target) + 1,
            help="Choose the variable you want to predict",
            key="target_selectbox"
        )
        
        if selected_target and selected_target != st.session_state.selected_target:
            st.session_state.selected_target = selected_target
            self.reset_downstream_selections(2)
        
        if not selected_features or not selected_target:
            st.sidebar.info("Please select at least one feature and one target variable")
            return
        
        st.sidebar.divider()
        
        # MISSING VALUES HANDLING
        st.sidebar.subheader("4️⃣ Handle Missing Values")
        selected_data = df[selected_features + [selected_target]]
        na_count = selected_data.isna().sum().sum()
        
        if na_count > 0:
            st.sidebar.warning(f"⚠️ {na_count} missing values detected")
            
            na_method = st.sidebar.selectbox(
                "Select filling method",
                options=["Select method...", "Delete rows", "Mean", "Median", "Constant"],
                key="na_method_select"
            )
            
            constant_value = None
            if na_method == "Constant":
                constant_value = st.sidebar.text_input(
                    "Enter constant value",
                    help="Value to replace missing data"
                )
            
            if st.sidebar.button("Apply NA Handling", type="primary", use_container_width=True):
                if na_method != "Select method...":
                    if na_method == "Constant" and not constant_value:
                        st.sidebar.error("Please enter a constant value")
                    else:
                        processed = self.handle_missing_values(
                            selected_data.copy(), 
                            na_method, 
                            constant_value
                        )
                        st.session_state.processed_data = processed
                        st.session_state.na_method = na_method
                        st.sidebar.success("✅ Missing values handled!")
                        st.rerun()
        else:
            st.sidebar.success("✅ No missing values detected")
            st.session_state.processed_data = selected_data
            st.session_state.na_method = "None needed"
        
        if st.session_state.processed_data is None:
            return
        
        st.sidebar.markdown("---")
        
        # TRAIN/TEST SPLIT
        st.sidebar.subheader("5️⃣ Train/Test Split")
        train_size = st.sidebar.slider(
            "Training Set Percentage",
            min_value=50,
            max_value=95,
            value=st.session_state.train_size,
            step=5,
            help="Percentage of data used for training (rest for testing)"
        )
        st.session_state.train_size = train_size
        
        total_rows = len(st.session_state.processed_data)
        train_rows = int(total_rows * train_size / 100)
        test_rows = total_rows - train_rows
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Train", f"{train_rows} rows")
        col2.metric("Test", f"{test_rows} rows")

    def render_main_content(self):
        """Render the main content area"""
        st.title("ModeLine")
        st.header("Train and visualize linear regression models")
        st.divider()
        
        if st.session_state.dataframe is None:
            st.info("👈 Please upload a dataset using the sidebar to get started")
            st.markdown("""
            ### Getting Started
            1. Upload your dataset (CSV, Excel, or SQLite)
            2. Select features for training (Only numeric)
            3. Choose your target variable (Numeric too)
            4. Handle missing values (if any)
            5. Configure train/test split
            6. Train your model!
            """)
            return
        
        df = st.session_state.dataframe
        cols_with_na = df.columns[df.isna().any()].tolist()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))
        col3.caption("**Columns with NA values**")
        col3.text((" | ").join(cols_with_na) if cols_with_na else "None")
        
        st.divider()
        
        # Display styled dataframe
        if st.session_state.selected_features or st.session_state.selected_target:
            st.markdown("#### Dataset Preview")
            st.caption("🔵 Blue columns = Features | 🔴 Red column = Target")
            styled_df = self.style_dataframe(df)
            st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            st.markdown("#### Dataset Preview")
            st.dataframe(df, use_container_width=True, height=400)
        
        # Show model training section
        if st.session_state.processed_data is not None:
            st.success("✅ Data is ready for training!")
            
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 TRAIN MODEL", type="primary", use_container_width=True):
                    with st.spinner("Training model..."):
                        # Prepare data
                        X = st.session_state.processed_data[st.session_state.selected_features]
                        y = st.session_state.processed_data[st.session_state.selected_target]
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y,
                            train_size=st.session_state.train_size / 100,
                            random_state=18
                        )
                        
                        # Train model
                        model = self.train_model(X_train, y_train)
                        
                        # Evaluate
                        metrics, y_train_pred, y_test_pred = self.evaluate_model(
                            model, X_train, y_train, X_test, y_test
                        )
                        
                        # Store in session state
                        st.session_state.model = model
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.y_train_pred = y_train_pred
                        st.session_state.y_test_pred = y_test_pred
                        st.session_state.metrics = metrics
                        st.session_state.model_trained = True
                        
                    st.rerun()
        
        # Display model results
        if st.session_state.model_trained:
            st.divider()
            st.header("📊 Model Results")
            
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("####  Training Set")
                metrics_train = st.session_state.metrics['train']
                st.metric("R² Score", f"{metrics_train['r2']:.4f}")
                st.metric("MSE", f"{metrics_train['mse']:.4f}")
            
            with col2:
                st.markdown("####  Test Set")
                metrics_test = st.session_state.metrics['test']
                st.metric("R² Score", f"{metrics_test['r2']:.4f}")
                st.metric("MSE", f"{metrics_test['mse']:.4f}")
            
            # Predictions plot
            st.divider()
            st.subheader("Predictions Visualization")
            fig = self.plot_predictions(
                st.session_state.y_train,
                st.session_state.y_train_pred,
                st.session_state.y_test,
                st.session_state.y_test_pred
            )
            st.plotly_chart(fig, use_container_width=True)

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
