from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import streamlit as st
import numpy as np


def predict():
    """Render a prediction UI and compute a single prediction.

    The function builds a small form that allows the user to input numeric
    values for each feature required by the trained or loaded model. The
    feature and target names are retrieved from **st.session_state** or the
    loaded packet. When the user clicks the predict button the model's
    **predict** method is called and the result is displayed.

    Side effects:
        - Reads **st.session_state** keys such as **features**, **target**,
          **model** and **loaded_packet**.
        - Renders Streamlit inputs and outputs (metrics, messages).

    Returns:
        None

    Raises:
        None: Errors during prediction are caught and displayed to the user
        via **st.error** rather than being raised.
    """

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

    inputs = np.array(
        [
            st.number_input(name, value=1.0, step=0.1, width=200)
            for name in features
        ]
    ).reshape(1, -1)

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
    """Linear Regression trainer with validation, splitting and evaluation.

    This class wraps common steps required to fit and evaluate a scikit-learn
    **LinearRegression** model: input validation, coercion to DataFrame,
    optional train/test splitting (disabled for very small datasets) and
    calculation of common metrics (R² and MSE).
    """

    def __init__(self, X, y, train_size, split_seed):
        """Initialize the trainer and prepare train/test subsets.

        Args:
            X (pandas.DataFrame or array-like): Feature matrix.
            y (pandas.DataFrame or array-like): Target vector or DataFrame.
            train_size (float): Fraction of data to use for training (0..1).
            split_seed (int): Random seed used for reproducible splitting.

        Raises:
            ValueError: If the inputs are invalid (mismatched lengths, empty,
                or train_size out of range).
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
        """Validate constructor inputs for basic consistency.

        Ensures that X and y are not None, have matching lengths, are not
        empty, and that **train_size** is within the [0, 1] range.

        Args:
            X: Feature matrix or equivalent.
            y: Target vector or equivalent.
            train_size (float): Train set fraction.

        Raises:
            ValueError: On any validation failure.
        """
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
        """Convert array-like inputs to pandas DataFrames.

        Args:
            X: Feature matrix (DataFrame or array-like).
            y: Target values (DataFrame or array-like).

        Returns:
            tuple: **(X_df, y_df)** where both are pandas DataFrames.
        """
        # Convert X to DataFrame if it's array-like
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Convert y to DataFrame if it's array-like
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        return X, y

    def _split_dataset(self, X, y, train_size, split_seed):
        """Split the dataset into training and test subsets.

        If the dataset is too small (fewer than 10 rows) the trainer will
        disable the test split and return the full dataset as the training
        set. Otherwise the function uses scikit-learn **train_test_split**
        with the provided **train_size** and **random_state**.

        Args:
            X (pandas.DataFrame): Feature DataFrame.
            y (pandas.DataFrame): Target DataFrame.
            train_size (float): Fraction of data for training (0..1).
            split_seed (int): Random seed for reproducibility.

        Returns:
            None: Subsets are stored on the instance as attributes
            **X_train**, **X_test**, **y_train** and **y_test**.
        """
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
        """Fit a scikit-learn LinearRegression model on the training data.

        Returns:
            sklearn.linear_model.LinearRegression: The fitted model instance.

        Raises:
            ValueError: If training data are not initialized.
            RuntimeError: If the underlying scikit-learn fit fails.
        """
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
        """Return the train/test subsets produced during initialization.

        Returns:
            tuple: **(X_train, X_test, y_train, y_test)**. For small datasets
            where no test split was created, **X_test** and **y_test** will
            be **None**.
        """
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_formula(self):
        """Return a human-readable regression formula string.

        The formula is constructed using feature names from **X_train** and
        the model coefficients and intercept. The resulting string is suitable
        for display in the UI.

        Returns:
            str: A formatted equation like **target = 0.1234 * x1 + ...**.

        Raises:
            ValueError: If the model has not been trained yet.
            RuntimeError: If coefficient extraction or formatting fails.
        """
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
        """Compute predictions and evaluate model performance.

        The method predicts on the training set and computes R² and MSE. If a
        test set is available (dataset large enough) it also predicts on the
        test set and computes the same metrics. The method returns a metrics
        dictionary together with the raw prediction arrays.

        Returns:
            tuple: **(metrics, y_train_pred, y_test_pred)** where **metrics**
            is a dict with keys **'train'** and optionally **'test'** and
            each contains **'r2'** and **'mse'** floats.

        Raises:
            ValueError: If called before training the model.
            RuntimeError: On prediction or metric computation errors.
        """
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
    

 
