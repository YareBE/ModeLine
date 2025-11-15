from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import streamlit as st


def predict():
        """Compute predictions when there exists a model (created or loaded)"""

        st.subheader("Predict with ModeLine")
        st.markdown("##### Enter the input values for your prediction")

        features = st.session_state.features

        new_features = []
        for name in features:
            new = st.number_input(name, value = 1)
        
            

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
        self.model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self._test_available = len(X) >= 10

        self._validate_inputs(X, y, train_size)
        X, y = self._ensure_dataframe(X, y)
        self._split_dataset(X, y, train_size, split_seed)

    @staticmethod
    def _validate_inputs(X, y, train_size):
        """Validate input parameters."""
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
            )
        if len(X) == 0:
            raise ValueError("X and y cannot be empty")
        if not (0 <= train_size <= 1):
            raise ValueError("train_size must be between 0 and 1")

    @staticmethod
    def _ensure_dataframe(X, y):
        """Convert to DataFrame if needed."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        return X, y

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

    def train_model(self):
        """Train the linear regression model."""
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Training data not initialized. Check __init__ method"
            )

        try:
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            return self.model
        except Exception as e:
            raise RuntimeError(f"Error training model: {str(e)}")

    def get_splitted_subsets(self):
        """Get train/test subsets."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_formula(self):
        """Get regression formula with feature names."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")

        try:
            feature_names = self.X_train.columns.tolist()
            target_name = self.y_train.columns[0]

            coefs = self.model.coef_.ravel()
            intercept = self.model.intercept_[0]

            formula_parts = [f"{target_name} = "]

            for i, (name, coef) in enumerate(zip(feature_names, coefs)):
                if i == 0:
                    formula_parts.append(f"{coef:.4f} * {name}")
                else:
                    sign = "+" if coef >= 0 else "-"
                    formula_parts.append(f" {sign} {abs(coef):.4f} * {name}")

            sign = "+" if intercept >= 0 else "-"
            formula_parts.append(f" {sign} {abs(intercept):.4f}")

            return "".join(formula_parts)

        except Exception as e:
            raise RuntimeError(f"Error generating formula: {str(e)}")

    def test_model(self):
        """Evaluate model performance on train/test sets."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")

        try:
            y_train_pred = self.model.predict(self.X_train)
            metrics = {
                'train': {
                    'r2': r2_score(self.y_train, y_train_pred),
                    'mse': mean_squared_error(self.y_train, y_train_pred)
                }
            }

            y_test_pred = None
            if self._test_available:
                y_test_pred = self.model.predict(self.X_test)
                metrics['test'] = {
                    'r2': r2_score(self.y_test, y_test_pred),
                    'mse': mean_squared_error(self.y_test, y_test_pred)
                }

            return metrics, y_train_pred, y_test_pred

        except Exception as e:
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

    def train_model(self):
        """Train the linear regression model."""
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Training data not initialized. Check __init__ method"
            )

        try:
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            return self.model
        except Exception as e:
            raise RuntimeError(f"Error training model: {str(e)}")

    def get_splitted_subsets(self):
        """Get train/test subsets."""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_formula(self):
        """Get regression formula with feature names."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")

        try:
            feature_names = self.X_train.columns.tolist()
            target_name = self.y_train.columns[0]

            coefs = self.model.coef_.ravel()
            intercept = self.model.intercept_[0]

            formula_parts = [f"{target_name} = "]

            for i, (name, coef) in enumerate(zip(feature_names, coefs)):
                if i == 0:
                    formula_parts.append(f"{coef:.4f} * {name}")
                else:
                    sign = "+" if coef >= 0 else "-"
                    formula_parts.append(f" {sign} {abs(coef):.4f} * {name}")

            sign = "+" if intercept >= 0 else "-"
            formula_parts.append(f" {sign} {abs(intercept):.4f}")

            return "".join(formula_parts)

        except Exception as e:
            raise RuntimeError(f"Error generating formula: {str(e)}")

    def test_model(self):
        """Evaluate model performance on train/test sets."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")

        try:
            y_train_pred = self.model.predict(self.X_train)
            metrics = {
                'train': {
                    'r2': r2_score(self.y_train, y_train_pred),
                    'mse': mean_squared_error(self.y_train, y_train_pred)
                }
            }

            y_test_pred = None
            if self._test_available:
                y_test_pred = self.model.predict(self.X_test)
                metrics['test'] = {
                    'r2': r2_score(self.y_test, y_test_pred),
                    'mse': mean_squared_error(self.y_test, y_test_pred)
                }

            return metrics, y_train_pred, y_test_pred

        except Exception as e:
            raise RuntimeError(f"Error testing model: {str(e)}")

  
