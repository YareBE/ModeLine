import pytest
import numpy as np
import io
import joblib
from sklearn.linear_model import LinearRegression
from model_serializer import packet_creation
import threading #Used for catching an exception 

@pytest.fixture
def model_mock():
    """Create a mock of a sklearn linear regression model"""
    model = LinearRegression()
    X_stub = np.array([[1, 2], [3, 4], [5, 6]])
    y_stub = np.array([4, 6, 8])
    model.fit(X_stub, y_stub)
    return model

def test_valid_packet(model_mock):
    """Test for creating a packet with valid parameters"""

    valid_buffer = packet_creation(
        model_mock, 
        "qwerty", 
        ["apple", "banana"], 
        ["grape"], 
        "e = mc2", 
        {"train":{'r':1, 'mse': 2}, "test":{'r':3, 'mse':4}}
    )
    assert isinstance(valid_buffer, io.BytesIO)

    valid_buffer.seek(0)
    try:
        joblib.load(valid_buffer)
    except Exception as e:
        pytest.fail(f"The buffer did not store a joblib file. Error: {e}")

def test_unexisting_model():
    """Test for creating a packet with no model"""
    with pytest.raises(ValueError, match = "Model must exist"):
        packet_creation(
        None, #
        "qwerty", 
        ["apple", "banana"], 
        ["grape"], 
        "e = mc2", 
        {"train":{'r':1, 'mse': 2}, "test":{'r':3, 'mse':4}}
    )
        
def test_unfitted_model():
    """Test for creating a packet with an unfitted model"""
    with pytest.raises(ValueError, match = "Model must be fitted"):
        packet_creation(
        LinearRegression(), #
        "qwerty", 
        ["apple", "banana"], 
        ["grape"], 
        "e = mc2", 
        {"train":{'r':1, 'mse': 2}, "test":{'r':3, 'mse':4}}
    )
        
def test_invalid_description(model_mock):
    """Test for creating a packet with an invalid description"""
    with pytest.raises(TypeError, match = "description must be"):
        packet_creation(
        model_mock, 
        18, #
        ["apple", "banana"], 
        ["grape"], 
        "e = mc2", 
        {"train":{'r':1, 'mse': 2}, "test":{'r':3, 'mse':4}}
    )

def test_unexisting_features(model_mock):
    """Test for creating a packet with no features"""
    with pytest.raises(ValueError, match = "cannot be None"):
        packet_creation(
        model_mock, 
        "qwerty", 
        [], #
        ["grape"], 
        "e = mc2", 
        {"train":{'r':1, 'mse': 2}, "test":{'r':3, 'mse':4}}
    )
        
def test_invalid_target_length(model_mock):
    """Test target validation: must be exactly one element"""
    with pytest.raises(ValueError, match="target must contain exactly 1 element"):
        packet_creation(
            model_mock,
            "qwerty",
            ["feature1"],
            ["target1", "target2"], #
            "e = mc2",
            {"train": {'r': 1}}
        )

def test_target_type_mismatch(model_mock):
    """Test target validation: must be a list, not a string"""
    with pytest.raises(TypeError, match="target must be list/tuple"):
        packet_creation(
            model_mock,
            "qwerty",
            ["feature1"],
            "target_string", #
            "e = mc2",
            {"train": {'r': 1}}
        )

def test_empty_formula(model_mock):
    """Test formula validation: cannot be empty or whitespace"""
    with pytest.raises(ValueError, match="formula cannot be empty string"):
        packet_creation(
            model_mock,
            "qwerty",
            ["feature1"],
            ["target1"],
            "   ",  #
            {"train": {'r': 1}}
        )

def test_invalid_formula(model_mock):
    """Test formula validation: must be a str"""
    with pytest.raises(TypeError, match="must be str"):
        packet_creation(
            model_mock,
            "qwerty",
            ["feature1"],
            ["target1"],
            33,  #
            {"train": {'r': 1}}
        )

def test_empty_metrics(model_mock):
    """Test metrics validation: cannot be an empty dict"""
    with pytest.raises(ValueError, match="metrics cannot be empty"):
        packet_creation(
            model_mock,
            "qwerty",
            ["feature1"],
            ["target1"],
            "formula",
            {}  #
        )

def test_serialization_failure(model_mock):
    """tries to serialize an 'unserializable' object"""
    trap_object = threading.Lock()
    corrupted = {"train": {'r': 1}, "test": trap_object}

    with pytest.raises(TypeError, match="Cannot serialize packet"):
        packet_creation(
            model_mock,
            "qwerty",
            ["feature1"],
            ["target1"],
            "formula",
            corrupted
        )



        











        