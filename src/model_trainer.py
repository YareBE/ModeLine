"""Linear Regression model training and prediction backend.

This module provides pure functions with no Streamlit dependencies,
enabling unit testing and backend separation from UI concerns.
All functions include comprehensive type hints and detailed docstrings.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import streamlit as st
import numpy as np
from typing import Tuple, Optional, Dict, Any


def modeline_prediction(model: LinearRegression, inputs: list) -> float:
    """Make a prediction using the trained model.
    
    Pure backend function with no Streamlit dependencies - suitable for
    unit testing and backend separation.
    
    Args:
        model (LinearRegression): Trained scikit-learn LinearRegression model.
        inputs (list): Feature values for prediction. Must match the number 
            of features the model was trained on.
    
    Returns:
        float: Single predicted value from the model.
    
    Raises:
        ValueError: If model is None, inputs empty, or shape mismatch.
        TypeError: If inputs cannot be converted to numpy array.
        RuntimeError: If prediction fails unexpectedly.
    """
    # Validate model exists
    if model is None:
        raise ValueError("Model cannot be None")
    
    # Validate inputs are not empty
    if inputs is None or (isinstance(inputs, (list, tuple)) and len(inputs) == 0):
        raise ValueError("Inputs cannot be empty")
    
    # Convert inputs to numpy array and reshape for prediction
    try:
        inputs_array = np.array(inputs).reshape(1, -1)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Cannot convert inputs to array: {str(e)}")
    
    # Validate input feature count matches model expectation
    expected_features = model.n_features_in_
    actual_features = inputs_array.shape[1]
    
    if actual_features != expected_features:
        raise ValueError(
            f"Input shape mismatch: expected {expected_features} "
            f"features, got {actual_features}"
        )
    
    # Perform prediction and return scalar
    try:
        prediction = model.predict(inputs_array)
        if prediction is None or len(prediction) == 0:
            raise RuntimeError("Model returned empty prediction")
        return float(prediction[0])
    except Exception as e:
        raise RuntimeError(f"Unexpected error during prediction: {str(e)}")


def validate_inputs(X: pd.DataFrame, y: pd.DataFrame, train_size: float) -> None:
    """Validate training data inputs for correctness and compatibility.
    
    Performs comprehensive validation of feature/target data including:
    - Null checks for X and y
    - Length compatibility (X and y must have same number of rows)
    - Empty data detection
    - train_size bounds validation (0..1 range)
    
    Args:
        X (pd.DataFrame): Feature DataFrame with training samples.
        y (pd.DataFrame): Target DataFrame with labels/values.
        train_size (float): Fraction of data allocated to training set (0..1).
    
    Returns:
        None (raises ValueError if validation fails).
    
    Raises:
        ValueError: If inputs are None, empty, length mismatch, or invalid train_size.
    """
    # Check for None values
    if X is None or y is None:
        raise ValueError("X and y cannot be None")
    
    # Check that X and y have same number of samples
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have same length. Got X: {len(X)}, y: {len(y)}"
        )
    
    # Check for empty data
    if len(X) == 0:
        raise ValueError("X and y cannot be empty")
    
    # Validate train_size is within valid range [0, 1]
    if not (0 <= train_size <= 1):
        raise ValueError("train_size must be between 0 and 1")


def ensure_dataframe(X: Any, y: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert array-like inputs to pandas DataFrames.
    
    Flexible input converter that accepts arrays, lists, or DataFrames
    and ensures output is always in DataFrame format for consistency.
    Useful for preprocessing varied input types before ML operations.
    
    Args:
        X (Any): Feature data (numpy array, list, or DataFrame).
        y (Any): Target data (numpy array, list, or DataFrame).
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of converted DataFrames (X, y).
    """
    # Convert X to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Convert y to DataFrame if needed
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y)
    
    return X, y


def split_dataset(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    train_size: float, 
    split_seed: int
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame]]:
    """Split dataset into training and test subsets.
    
    Splits feature and target data for model training/evaluation using sklearn's
    train_test_split. For very small datasets (<10 samples), returns full data 
    for training and None for test sets to avoid edge cases.
    
    Args:
        X (pd.DataFrame): Feature DataFrame with n_samples rows.
        y (pd.DataFrame): Target DataFrame with n_samples rows.
        train_size (float): Fraction of data for training (0..1).
        split_seed (int): Random seed for reproducible splits.
    
    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame]]:
            - X_train: Training feature data
            - X_test: Test feature data (None if dataset too small)
            - y_train: Training target data
            - y_test: Test target data (None if dataset too small)
    
    Raises:
        ValueError: If validation fails or seed is invalid.
        TypeError: If inputs are not DataFrames.
        RuntimeError: If split operation fails.
    """
    # Ensure inputs are DataFrames
    X, y = ensure_dataframe(X, y)
    
    # Validate inputs
    validate_inputs(X, y, train_size)
    
    # Validate split_seed is a valid non-negative integer
    try:
        split_seed = int(split_seed)
        if split_seed < 0:
            raise ValueError("split_seed must be non-negative")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid split_seed: {str(e)}")
    
    # Check dataset size - skip split for tiny datasets
    n_samples = len(X)
    if n_samples < 10:
        # For very small datasets, use all data for training
        return X, None, y, None
    
    # Perform train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=split_seed
        )
        
        # Validate split produced non-empty subsets
        if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
            raise RuntimeError("Split resulted in empty subsets")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise RuntimeError(f"Error during train/test split: {str(e)}")


def train_linear_regression(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame
) -> LinearRegression:
    """Train a LinearRegression model on training data.
    
    Fits a scikit-learn LinearRegression model to the provided training data.
    Validates input data before fitting and checks model coefficients after training.
    Returns the fitted model ready for predictions.
    
    Args:
        X_train (pd.DataFrame): Training feature data with n_samples rows.
        y_train (pd.DataFrame): Training target data with n_samples rows.
    
    Returns:
        LinearRegression: Trained scikit-learn LinearRegression model with 
            fitted coefficients and intercept ready for predictions.
    
    Raises:
        ValueError: If X_train or y_train is None or validation fails.
        TypeError: If inputs cannot be converted to DataFrames.
        RuntimeError: If model fitting fails.
    """
    # Check for None inputs
    if X_train is None:
        raise ValueError("X_train cannot be None")
    if y_train is None:
        raise ValueError("y_train cannot be None")
    
    # Ensure inputs are DataFrames
    X_train, y_train = ensure_dataframe(X_train, y_train)
    
    # Validate inputs (use dummy train_size=0.8 for validation purposes)
    validate_inputs(X_train, y_train, train_size=0.8)
    
    # Fit LinearRegression model
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Validate model was fitted correctly
        if model.coef_ is None or model.intercept_ is None:
            raise RuntimeError("Model fit failed: coefficients not computed")
        
        return model
    except ValueError as e:
        raise ValueError(f"Model fitting validation failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error training LinearRegression: {str(e)}")


def generate_formula(
    model: LinearRegression, 
    feature_names: list, 
    target_name: str
) -> str:
    """Generate human-readable regression formula from trained model.
    
    Creates a readable formula string representation of the linear regression
    equation in the mathematical form: target = coef1*feature1 + coef2*feature2 + ... + bias
    Coefficients are formatted to 4 decimal places for readability.
    
    Args:
        model (LinearRegression): Trained scikit-learn LinearRegression model.
        feature_names (list): List of feature column names in order.
        target_name (str): Name of target variable.
    
    Returns:
        str: Human-readable formula string with formatted coefficients (4 decimals).
    
    Raises:
        ValueError: If model is None.
        RuntimeError: If formula generation fails.
    """
    # Validate model
    if model is None:
        raise ValueError("Model cannot be None")

    try:
        # Extract coefficients and intercept
        coefs = model.coef_.ravel()
        intercept = (
            model.intercept_[0] 
            if isinstance(model.intercept_, np.ndarray) 
            else model.intercept_
        )

        # Build formula parts
        formula_parts = [f"{target_name} = "]

        # Add each feature coefficient
        for i, (name, coef) in enumerate(zip(feature_names, coefs)):
            if i == 0:
                # First term doesn't need sign prefix
                formula_parts.append(f"{coef:.4f} * {name}")
            else:
                # Subsequent terms include +/- sign
                sign = "+" if coef >= 0 else "-"
                formula_parts.append(f" {sign} {abs(coef):.4f} * {name}")

        # Add intercept term
        sign = "+" if intercept >= 0 else "-"
        formula_parts.append(f" {sign} {abs(intercept):.4f}")

        return "".join(formula_parts)

    except Exception as e:
        raise RuntimeError(f"Error generating formula: {str(e)}")


def evaluate_model(
    model: LinearRegression, 
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None, 
    y_test: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Dict[str, float]]]:
    """Evaluate model performance on training and optional test data.
    
    Computes R² (coefficient of determination) and MSE (mean squared error) metrics.
    Always returns training metrics; test metrics included only if test data provided.
    Returns model predictions alongside metrics for further analysis.
    
    Args:
        model (LinearRegression): Trained scikit-learn LinearRegression model.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.DataFrame): Training target data.
        X_test (Optional[pd.DataFrame]): Test feature data (default: None).
        y_test (Optional[pd.DataFrame]): Test target data (default: None).
    
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Dict[str, float]]]:
            - y_train_pred: Model predictions on training data (1D array)
            - y_test_pred: Model predictions on test data (None if no test set)
            - metrics: Dict with structure:
                {
                    'train': {'r2': float, 'mse': float},
                    'test': {'r2': float, 'mse': float}  # Only if test set provided
                }
    
    Raises:
        ValueError: If model is None.
        RuntimeError: If evaluation fails.
    """
    # Validate model
    if model is None:
        raise ValueError("Model cannot be None")

    try:
        # Get training predictions
        y_train_pred = model.predict(X_train)

        # Calculate training metrics
        metrics = {
            'train': {
                'r2': float(r2_score(y_train, y_train_pred)),
                'mse': float(mean_squared_error(y_train, y_train_pred))
            }
        }

        # Calculate test metrics if test set provided
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            metrics['test'] = {
                'r2': float(r2_score(y_test, y_test_pred)),
                'mse': float(mean_squared_error(y_test, y_test_pred))
            }
            return y_train_pred, y_test_pred, metrics
        
        # Return None for test predictions if no test set
        return y_train_pred, None, metrics

    except Exception as e:
        raise RuntimeError(f"Error evaluating model: {str(e)}")
 
