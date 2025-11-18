from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import streamlit as st
import numpy as np


def predict():
        """Compute predictions when there exists a model (created or loaded)"""

        st.subheader("Predict with ModeLine")
        st.markdown("##### Enter the input values for your prediction")

        features = st.session_state.features or st.session_state.loaded_packet.get("features")
        target = st.session_state.target or st.session_state.loaded_packet.get("target")

        if not features or not target:
            st.error("Features or target not found in session state")
            return  
        
        model = st.session_state.get("model") or st.session_state.loaded_packet.get("model")
        if model is None:
            st.error("No model found. Train or load a model first")
            return

        inputs = np.array([
            st.number_input(name, value=1.0, step=0.1, width=200) 
            for name in features]).reshape(1, -1)

        if st.button("PREDICT", type="primary"):
            try:
                prediction = model.predict(inputs)
                    
                st.markdown("#### Predicted Value:")
                
                # Display results
                st.metric(
                    label=f"Predicted {target}",
                    value=f"{prediction[0][0]:,.3f}"
                )
                    
                st.success("✅ Prediction completed successfully!")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

class LRTrainer:
    """Linear Regression trainer with validation and evaluation."""

    def __init__(self, X, y, train_size, split_seed):
        """Initialize Linear Regression trainer.

        Args:
            X: Features (DataFrame or array-like)
            y: Target variable (DataFrame or array-like)
            train_size: Training set proportion (0-1)
            split_seed: Random seed for reproducibility
        """
        # Initialize model and prediction attributes
        self.model = None
        self.y_train_pred = None
        self.y_test_pred = None
        # Determine if dataset is large enough for train/test split
        # Minimum 10 rows required
        self._test_available = len(X) >= 10
        
        # Validate inputs before processing
        self._validate_inputs(X, y, train_size)
        # Ensure data is in DataFrame format for consistent handling
        X, y = self._ensure_dataframe(X, y)
        # Split data into train/test sets
        self._split_dataset(X, y, train_size, split_seed)

    @staticmethod
    def _validate_inputs(X, y, train_size):
        """Validate input parameters."""
        # Check for None inputs
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        # Validate matching dimensions
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
            )
        
        # Check for empty datasets
        if len(X) == 0:
            raise ValueError("X and y cannot be empty")

        # Validate train size is a valid proportion
        if not (0 <= train_size <= 1):
            raise ValueError("train_size must be between 0 and 1")

    @staticmethod
    def _ensure_dataframe(X, y):
        """Convert to DataFrame if needed."""
        # Convert X to DataFrame if it's array-like
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Convert y to DataFrame if it's array-like
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        return X, y

    def _split_dataset(self, X, y, train_size, split_seed):
        """Split dataset into train and test sets."""
        if self._test_available:
            # Standard train/test split for adequate dataset size
            (self.X_train, self.X_test, self.y_train,
             self.y_test) = train_test_split(
                X, y, train_size=train_size, random_state=split_seed
            )
        else:
            # Use all data for training, no test set
            self.X_train, self.X_test, self.y_train, self.y_test = (
                X, None, y, None
            )

    def train_model(self):
        """Train the linear regression model."""
        # Validate training data exists 
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Training data not initialized. Check __init__ method"
            )

        try:
            # Initialize sklearn's LinearRegression model
            self.model = LinearRegression()

            # Fit model using training data (computes coefficients)
            self.model.fit(self.X_train, self.y_train)

            return self.model
        except Exception as e:
            # Catch fitting errors 
            raise RuntimeError(f"Error training model: {str(e)}")

    def get_splitted_subsets(self):
        """Get train/test subsets."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_formula(self):
        """Get regression formula with feature names."""
        # Ensure model is trained before accessing coefficients
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")

        try:
            # Extract feature and target names from DataFrames
            feature_names = self.X_train.columns.tolist()
            target_name = self.y_train.columns[0]

            # Get model coefficients and intercept
            coefs = self.model.coef_.ravel()
            intercept = self.model.intercept_[0]

            # Build formula string piece by piece
            formula_parts = [f"{target_name} = "]

            # Add coefficient terms for each feature
            for i, (name, coef) in enumerate(zip(feature_names, coefs)):
                if i == 0:
                    # First term: no leading +/- sign
                    formula_parts.append(f"{coef:.4f} * {name}")
                else:
                    # Subsequent terms: add explicit +/- sign
                    sign = "+" if coef >= 0 else "-"
                    formula_parts.append(f" {sign} {abs(coef):.4f} * {name}")

            # Add intercept term with appropriate sign
            sign = "+" if intercept >= 0 else "-"
            formula_parts.append(f" {sign} {abs(intercept):.4f}")

            # Join all parts into single string
            return "".join(formula_parts)

        except Exception as e:
            raise RuntimeError(f"Error generating formula: {str(e)}")

    def test_model(self):
        """Evaluate model performance on train/test sets."""
        # Ensure model is trained before making predictions
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")

        try:
            # Generate predictions for training set
            y_train_pred = self.model.predict(self.X_train)

            # Calculate training metrics
            metrics = {
                'train': {
                    'r2': r2_score(self.y_train, y_train_pred),
                    'mse': mean_squared_error(self.y_train, y_train_pred)
                }
            }

            # Initialize test predictions as None (for small datasets)
            y_test_pred = None

            # Calculate test metrics if test set exists
            if self._test_available:
                y_test_pred = self.model.predict(self.X_test)
                metrics['test'] = {
                    'r2': r2_score(self.y_test, y_test_pred),
                    'mse': mean_squared_error(self.y_test, y_test_pred)
                }

            return metrics, y_train_pred, y_test_pred

        except Exception as e:
            # Catch prediction or metric calculation errors
            raise RuntimeError(f"Error testing model: {str(e)}")

    def _split_dataset(self, X, y, train_size, split_seed):
        """Split dataset into train and test sets."""
        if self._test_available:
            (self.X_train, self.X_test, self.y_train,
             self.y_test) = train_test_split(
                X, y, train_size=train_size, random_state=split_seed
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                X, None, y, None
            )

 
