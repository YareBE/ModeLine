"""Model serialization and packet creation for trained models.

This module handles packaging trained LinearRegression models with metadata
(description, features, target, formula, metrics) into joblib-serialized packets
for downloading and later reuse.
"""

import io
import joblib
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted, NotFittedError


def packet_creation(
    model: LinearRegression,
    description: str,
    features: List[str],
    target: List[str],
    formula: str,
    metrics: Dict[str, Dict[str, float]]
) -> io.BytesIO:
    """Create a joblib-serializable packet with model and metadata.

    Pure backend function with no Streamlit dependencies - validates all inputs
    thoroughly and creates an in-memory binary buffer containing the serialized
    model packet ready for download or storage.

    Args:
        model (LinearRegression): Trained scikit-learn LinearRegression model
            with fitted coefficients and intercept.
        description (str): User-provided model description (can be empty).
        features (List[str]): List of feature column names used for training.
        target (List[str]): List containing exactly one target variable name.
        formula (str): Human-readable model formula string representation.
        metrics (Dict[str, Dict[str, float]]): Performance metrics dict with structure:
            {
                'train': {'r2': float, 'mse': float},
                'test': {'r2': float, 'mse': float}  # Optional
            }

    Returns:
        io.BytesIO: In-memory buffer containing joblib-serialized packet dict.
            Buffer is positioned at start (seek(0)) ready for reading/download.

    Raises:
        ValueError: If model is None/unfitted, features/target empty, or invalid content.
        TypeError: If parameters have incorrect types.
        RuntimeError: If serialization to joblib format fails.
    """
    # Validate all inputs
    validate_model(model)
    validate_description(description)
    validate_features(features)
    validate_target(target)
    validate_formula(formula)
    validate_metrics(metrics)

    # Build packet dictionary with model and metadata
    try:
        packet = {
            "model": model,
            "description": description,
            "features": list(features),
            "target": list(target),
            "formula": formula,
            "metrics": metrics,
            "app": "ModeLine"
        }
    except Exception as e:
        raise RuntimeError(f"Error building packet dictionary: {str(e)}")

    # Serialize packet to in-memory binary buffer
    try:
        buffer = io.BytesIO()
        joblib.dump(packet, buffer)
        buffer.seek(0)  # Position at start for reading

        # Validate buffer is not empty
        buffer_size = buffer.getbuffer().nbytes
        if buffer_size == 0:
            raise RuntimeError("Serialization produced empty buffer")

        return buffer
    except TypeError as e:
        raise TypeError(
            f"Cannot serialize packet (non-serializable object): {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error serializing packet to joblib: {str(e)}")


def validate_model(model: LinearRegression) -> None:
    """Validate the LinearRegression model"""
    if model is not None and isinstance(model, LinearRegression):
        try:
            check_is_fitted(model)
        except NotFittedError:
            raise ValueError("Model must be fitted")
    else:
        raise ValueError("Model must exist")


def validate_description(description: str) -> None:
    """Validate the description parameter"""
    if description is None:
        description = ""
    elif not isinstance(description, str):
        raise TypeError(
            f"description must be str, got "
            f"{type(description).__name__}")


def validate_features(features: List[str]) -> None:
    """Validate the features parameter"""
    if features is None or (
        isinstance(features, (list, tuple)) and len(features) == 0
    ):
        raise ValueError("features cannot be None or empty")
    if not isinstance(features, (list, tuple)):
        raise TypeError(
            f"features must be list/tuple, got {type(features).__name__}")
    if not all(isinstance(f, str) for f in features):
        raise TypeError("All features must be strings")


def validate_target(target: List[str]) -> None:
    """Validate the target parameter"""
    if target is None or (isinstance(target, (list, tuple))
                          and len(target) == 0):
        raise ValueError("target cannot be None or empty")
    if not isinstance(target, (list, tuple)):
        raise TypeError(
            f"target must be list/tuple, got {type(target).__name__}")
    if len(target) != 1:
        raise ValueError(
            f"target must contain exactly 1 element, got "
            f"{len(target)}")
    if not isinstance(target[0], str):
        raise TypeError("target element must be string")


def validate_formula(formula: str) -> None:
    """Validate the formula parameter"""
    if formula is None:
        raise ValueError("formula cannot be None")
    if not isinstance(formula, str):
        raise TypeError(f"formula must be str, got {type(formula).__name__}")
    if len(formula.strip()) == 0:
        raise ValueError("formula cannot be empty string")


def validate_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Validate the metrics parameter"""
    if metrics is None:
        raise ValueError("metrics cannot be None")
    if not isinstance(metrics, dict):
        raise TypeError(f"metrics must be dict, got {type(metrics).__name__}")
    if len(metrics) == 0:
        raise ValueError("metrics cannot be empty")
