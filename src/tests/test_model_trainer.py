import model_trainer as trainer
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.utils.validation import NotFittedError
import pytest


# --- FIXTURES ---

@pytest.fixture
def model_mock():
    """Create a mock of a sklearn linear regression model"""
    model = LinearRegression()
    X_stub = np.array([[1, 2], [3, 4], [5, 6]])
    y_stub = np.array([3, 7, 11])
    model.fit(X_stub, y_stub)
    return model

@pytest.fixture
def small_dataset_mock():
    """
    Create a simple mock of a dataset with linearly independent features.
    Formula: pizza = 2 * apples + 1 * oranges + 0 (intercept)
    """
    X = pd.DataFrame([
        [1, 0], 
        [2, 1], 
        [3, 0]
    ], columns=['apples', 'oranges'])
    
    y = pd.DataFrame([2, 5, 6], columns=['pizza'])
    return X, y

@pytest.fixture
def big_dataset_mock():
    """Create a larger mock of a dataset"""
    X = pd.DataFrame({'apples':[p for p in range(15)], 
                    'oranges':[q*2 for q in range(15)]})
    y = pd.DataFrame({'pizzas': [r*3 for r in range(15)]})
    return X, y


# --- TEST PREDICTIONS ---

def test_correct_predictions(model_mock):
    """Testing if the function predicts correctly"""
    # Adjusted assertions to match sklearn output format (numpy arrays)
    prediction = trainer.modeline_prediction(model_mock, [4, 4.5])
    assert np.allclose(prediction, [[8.5]])
    
    prediction_zero = trainer.modeline_prediction(model_mock, [0, 0])
    assert np.allclose(prediction_zero, [[0]])

def test_prediction_no_model():
    """Test for trying to predict with no existing model"""
    with pytest.raises(ValueError, match="Model must exist"):
        trainer.modeline_prediction(None, [10, 20])

def test_prediction_unfitted_model():
    """Test for trying to predict with an unfitted model"""
    with pytest.raises(NotFittedError, match="Model must be fitted"):
        trainer.modeline_prediction(LinearRegression(), [10, 20])

def test_prediction_no_inputs(model_mock):
    """Test for trying to predict with no inputs"""
    with pytest.raises(ValueError, match="Inputs cannot be empty"):
        trainer.modeline_prediction(model_mock, [])

def test_prediction_different_features_length(model_mock):
    """Test for trying to predict with an invalid number of features"""
    with pytest.raises(ValueError, match="shape mismatch"):
        trainer.modeline_prediction(model_mock, [1, 4, 9])


# --- TEST INPUT VALIDATION ---

@pytest.mark.parametrize("unexisting_inputs", [
    (None, None), 
    (1, None),
    (None, 1)              
])
def test_inputs_validation_no_parameters(unexisting_inputs):
    """Test for validating with no features or target"""
    with pytest.raises(ValueError, match="cannot be None"):
        X, y = unexisting_inputs
        trainer.validate_inputs(X, y, 0.5)

def test_inputs_validation_different_lengths():
    """Test for validating with not the same length of features and target"""
    with pytest.raises(ValueError, match="must have same length"):
        trainer.validate_inputs(pd.DataFrame([1, 2]), pd.DataFrame([5]), 0.5)

def test_inputs_validation_empty_inputs():
    """Test for validating with empty datasets"""
    with pytest.raises(ValueError, match="cannot be empty"):
        trainer.validate_inputs(pd.DataFrame([]), pd.DataFrame([]), 0.5)

@pytest.mark.parametrize("invalid_training_size", [
    -2,
    1.5             
])
def test_inputs_validation_invalid_training_size(invalid_training_size, small_dataset_mock):
    """Test for validating with invalid training sizes"""
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        X, y = small_dataset_mock
        trainer.validate_inputs(X, y, invalid_training_size)

@pytest.mark.parametrize("valid_training_size", [
    0.1,
    0.5,
    0.9            
])
def test_inputs_validation_proper_cases(small_dataset_mock, valid_training_size):
    """Test for validating inputs with correct cases"""
    X, y = small_dataset_mock
    trainer.validate_inputs(X, y, valid_training_size)


# --- TEST DATAFRAME CONVERTER ---

@pytest.mark.parametrize("invalid_dataframe_types", [
    ("X", [5, 3]),
    ("X", 7),
    ([], 3)     
])
def test_dataframe_converter_invalid_types(invalid_dataframe_types):
    """Test for capturing invalid 'array-like' inputs"""
    X, y = invalid_dataframe_types
    with pytest.raises(TypeError, match="was not possible"):
        trainer.ensure_dataframe(X, y)

@pytest.mark.parametrize("proper_dataframe_types", [
    ([1], [3]),
    (pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), pd.Series([5, 3])), 
    ({'un': [10]}, {'de': ['Y']})        
])
def test_dataframe_converter_proper_types(proper_dataframe_types):
    """Test for capturing valid inputs and converting them"""
    # Unpack the tuple from parameters
    input_X, input_y = proper_dataframe_types
    X, y = trainer.ensure_dataframe(input_X, input_y)
    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame)


# --- TEST DATASET SPLIT ---

@pytest.mark.parametrize("invalid_seed_types", [
    [3],
    {"a":4},
    LinearRegression()   
])
def test_dataset_split_invalid_seed_type(small_dataset_mock, invalid_seed_types):
    """Test for capturing invalid types of seed"""
    X, y = small_dataset_mock
    with pytest.raises(TypeError, match="Invalid split_seed type"):
        trainer.split_dataset(X, y, 0.8, invalid_seed_types)

@pytest.mark.parametrize("invalid_seed_values", [
    -1,
    -2,
    -4.5
])
def test_dataset_split_invalid_seed_values(small_dataset_mock, invalid_seed_values):
    """Test for capturing invalid values of seed (negative)"""
    X, y = small_dataset_mock
    with pytest.raises(ValueError, match="must be non-negative"):
        trainer.split_dataset(X, y, 0.8, invalid_seed_values)

@pytest.mark.parametrize("proper_seed", [3, 4, 1.2]) 
def test_dataset_split_full_train_case(small_dataset_mock, proper_seed):
    """
    Test for splitting when train_size is 1 (Return full dataset).
    Checks that X_test/y_test are None/Empty based on your logic.
    """
    X, y = small_dataset_mock
    
    # We force train_size=1.0 to verify logic where no test set is created
    final_X, null_X, final_y, null_y = trainer.split_dataset(X, y, 1.0, proper_seed)

    pd.testing.assert_frame_equal(X, final_X)
    pd.testing.assert_frame_equal(y, final_y)
    assert null_X is None
    assert null_y is None

def test_split_reproducibility(big_dataset_mock):
    """Test for checking if the splitting is repeateable"""
    X, y = big_dataset_mock
    
    X_tr1, X_te1, y_tr1, y_te1 = trainer.split_dataset(X, y, 0.8, 18)
    X_tr2, X_te2, y_tr2, y_te2 = trainer.split_dataset(X, y, 0.8, 18)
    
    np.testing.assert_array_equal(X_tr1, X_tr2)
    np.testing.assert_array_equal(y_te1, y_te2)

def test_split_randomness(big_dataset_mock):
    """Different seed must produce different results"""
    X, y = big_dataset_mock
    X_tr_A, _, _, _ = trainer.split_dataset(X, y, 0.8, 6)
    X_tr_B, _, _, _ = trainer.split_dataset(X, y, 0.8, 18)

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(X_tr_A, X_tr_B)


# --- TEST TRAINING ---

def test_train_linear_regression_success(big_dataset_mock):
    """Tests whether a correct set of inputs returns a LinearRegression model"""
    X_train, y_train = big_dataset_mock

    model = trainer.train_linear_regression(X_train, y_train)

    assert isinstance(model, LinearRegression)
    assert hasattr(model, 'coef_'), "The model should have 'coef_' attribute"
    assert hasattr(model, 'intercept_'), "The model should have 'intercept_' attribute"
    assert model.coef_ is not None, "Coefficients should not be None"

def test_train_linear_regression_non_numeric_error():
    """Sad Path: Verifies that providing DataFrames with non-numeric
    types raises a ValueError"""

    X_train_bad = pd.DataFrame({
        'feature1': ["apple", "banana", "cherry"], 
        'feature2': ["dog", "cat", "bird"]
    })
    y_train_bad = pd.DataFrame({
        'target': [1, 2, 3]
    })

    with pytest.raises(ValueError, match="Model fitting validation failed"):
        trainer.train_linear_regression(X_train_bad, y_train_bad)


# --- TEST FORMULA GENERATION ---

def test_generate_formula_success(small_dataset_mock):
    """Checks if the formula returned is coherent with the given model"""
    X, y = small_dataset_mock
    model = LinearRegression()
    model.fit(X, y)

    formula = trainer.generate_formula(model, list(X.columns), "pizza") 
    expected_formula = "pizza = 2.0000 * apples + 1.0000 * oranges + 0.0000"
    
    assert formula == expected_formula

def test_generate_formula_none_model():
    """Verify that no passing a model raises a ValueError"""
    with pytest.raises(ValueError, match="Model cannot be None"):
        trainer.generate_formula(None, ["f1"], "target")

def test_generate_formula_unfitted_error():
    """Verify that trying to generate a formula of an unfitted model results
    in a runtime error"""
    unfitted_model = LinearRegression()
    
    with pytest.raises(RuntimeError, match="Error generating formula"):
        trainer.generate_formula(unfitted_model, ["f1"], "target")


# --- TEST EVALUATION ---

def test_evaluate_model_full_flow(big_dataset_mock, small_dataset_mock):
    """Test the functionality with valid inputs"""
    dummy_model = LinearRegression()
    X_train, y_train = big_dataset_mock
    X_test, y_test = small_dataset_mock
    dummy_model.fit(X_train, y_train)

    y_tr_pred, y_te_pred, metrics = trainer.evaluate_model(
        dummy_model, X_train, y_train, X_test, y_test
    )

    assert isinstance(y_tr_pred, np.ndarray)
    assert isinstance(y_te_pred, np.ndarray)

    # It contains both Keys
    assert 'train' in metrics
    assert 'test' in metrics
    
    # Check content of metrics (metrics must be floats)
    assert isinstance(metrics['train']['mse'], float)
    assert isinstance(metrics['test']['r2'], float)

def test_evaluate_model_train_only(model_mock, small_dataset_mock):
    """Check if evaluating a small dataset's model returns metrics with only
    parameters w.r.t the training set"""
    
    X_train, y_train = small_dataset_mock
    
    y_tr_pred, y_te_pred, metrics = trainer.evaluate_model(
        model_mock, X_train, y_train
    )

    assert y_te_pred is None
    assert 'train' in metrics
    assert 'test' not in metrics

def test_evaluate_model_dimension_mismatch(model_mock):
    """Verify if passing dataframes with different sizes raises an exception"""
    X_broken = pd.DataFrame({'f1': [1, 2]}) 
    y_broken = pd.DataFrame({'t': [10, 20, 30]})

    with pytest.raises(RuntimeError, match="Error evaluating model"):
        trainer.evaluate_model(model_mock, X_broken, y_broken)