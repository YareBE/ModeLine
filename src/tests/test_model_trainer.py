import pytest
from src.model_trainer import *

# Prueba del constructor
def test_init_valid(sample_data):
    X, y = sample_data
    trainer = LRTrainer(X, y, train_size=0.8, split_seed=42)

    assert len(trainer.X_train) > 0
    assert trainer.model is None


def test_init_invalid_lengths():
    X = [[1], [2], [3]]
    y = [[1], [2]]   # Longitudes diferentes

    with pytest.raises(ValueError):
        LRTrainer(X, y, 0.8, 42)


# Prueba del entrenador del modelo
def test_training(sample_data):
    X, y = sample_data
    trainer = LRTrainer(X, y, 0.8, 42)
    model = trainer.train_model()

    assert model is not None
    assert model.coef_ is not None


# Pruebas de la formula generada
def test_formula(sample_data):
    X, y = sample_data
    trainer = LRTrainer(X, y, 0.8, 42)
    trainer.train_model()

    formula = trainer.get_formula()

    assert isinstance(formula, str)
    assert "target" in formula
    assert "feat1" in formula


# Pruebas de la evaluacion de modelo
def test_metrics(sample_data):
    X, y = sample_data
    trainer = LRTrainer(X, y, 0.8, 42)
    trainer.train_model()
    metrics, train_pred, test_pred = trainer.test_model()

    assert "train" in metrics
    assert "r2" in metrics["train"]
    assert "mse" in metrics["train"]
