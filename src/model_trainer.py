"""Linear Regression model training and prediction backend.

Pure functions with no Streamlit dependencies for unit testing.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import streamlit as st
import numpy as np


def predict():
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
            prediction = _modeline_prediction(model, inputs)
            st.markdown(f"#### Predicted {target[0]}: {prediction:,.3f}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")


def _modeline_prediction(model, inputs): ##
    """Make a prediction using the trained model (backend)."""
    if model is None:
        raise ValueError("Model cannot be None")
    
    if inputs is None or (isinstance(inputs, (list, tuple)) and len(inputs) == 0):
        raise ValueError("Inputs cannot be empty")
    
    try:
        inputs_array = np.array(inputs).reshape(1, -1)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot convert inputs to array: {str(e)}")
    
    expected_features = model.n_features_in_
    actual_features = inputs_array.shape[1]
    
    if actual_features != expected_features:
        raise ValueError(
            f"Input shape mismatch: expected {expected_features} "
            f"features, got {actual_features}"
        )
    
    try:
        prediction = model.predict(inputs_array)
        if prediction is None or len(prediction) == 0:
            raise RuntimeError("Model returned empty prediction")
        return float(prediction[0])
    except Exception as e:
        raise RuntimeError(f"Unexpected error during prediction: {str(e)}")


def _validate_inputs(X, y, train_size): ##
    """Validate inputs (backend)."""
    if X is None or y is None:
        raise ValueError("X and y cannot be None")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
        )
    if len(X) == 0:
        raise ValueError("X and y cannot be empty")
    if not (0 < train_size < 1):
        raise ValueError("train_size must be between 0 and 1 (exclusive)")


def _ensure_dataframe(X, y): ##
    """Convert array-like inputs to pandas DataFrames (backend)."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y)
    return X, y


def split_dataset(X, y, train_size, split_seed): ##
    """Split the dataset into training and test subsets (backend)."""
    # Ensure inputs are DataFrames
    X, y = _ensure_dataframe(X, y)
    
    # Validate inputs
    _validate_inputs(X, y, train_size)
    
    # Validate split_seed
    try:
        split_seed = int(split_seed)
        if split_seed < 0:
            raise ValueError("split_seed must be non-negative")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid split_seed: {str(e)}")
    
    # Check dataset size
    n_samples = len(X)
    if n_samples < 10:
        return X, None, y, None
    
    # Perform split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=split_seed
        )
        if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
            raise RuntimeError("Split resulted in empty subsets")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise RuntimeError(f"Error during train/test split: {str(e)}")


def train_linear_regression(X_train, y_train): ##
    """Train a LinearRegression model on training data (backend)."""
    if X_train is None:
        raise ValueError("X_train cannot be None")
    if y_train is None:
        raise ValueError("y_train cannot be None")
    
    # Ensure inputs are DataFrames
    X_train, y_train = _ensure_dataframe(X_train, y_train)
    
    # Validate inputs (use dummy train_size=0.8 for validation purposes)
    _validate_inputs(X_train, y_train, train_size=0.8)
    
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        if model.coef_ is None or model.intercept_ is None:
            raise RuntimeError("Model fit failed: coefficients not computed")
        return model
    except ValueError as e:
        raise ValueError(f"Model fitting validation failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error training LinearRegression: {str(e)}")


def generate_formula(model, feature_names, target_name): ##
    """Generate a human-readable regression formula string (backend)."""
    if model is None:
        raise ValueError("Model cannot be None")

    try:
        coefs = model.coef_.ravel()
        intercept = model.intercept_[0] if isinstance(model.intercept_, np.ndarray) else model.intercept_

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


def evaluate_model(model, X_train, y_train, X_test=None, y_test=None): ##
    """Evaluate model performance on train/test sets (backend)."""
    if model is None:
        raise ValueError("Model cannot be None")

    try:
        y_train_pred = model.predict(X_train)

        metrics = {
            'train': {
                'r2': float(r2_score(y_train, y_train_pred)),
                'mse': float(mean_squared_error(y_train, y_train_pred))
            }
        }

        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            metrics['test'] = {
                'r2': float(r2_score(y_test, y_test_pred)),
                'mse': float(mean_squared_error(y_test, y_test_pred))
            }
            return y_train_pred, y_test_pred, metrics
        
        return y_train_pred, None, metrics

        

    except Exception as e:
        raise RuntimeError(f"Error evaluating model: {str(e)}")
 
