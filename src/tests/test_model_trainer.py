from model_trainer import modeline_prediction
from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st
import numpy as np
import pytest
from model_trainer import LinearRegression


class TestLRTrainerTrainModel:
    """Tests para train_model() de LRTrainer."""

    def test_trains_model_successfully(self, clean_df):
        """Debe entrenar modelo correctamente."""
        X = clean_df[['x1', 'x2']]
        y = clean_df[['y']]
        trainer = LinearRegression(X, y, train_size=0.8, split_seed=42)

        model = trainer.train_model()

        assert model is not None
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert trainer.model == model

    def test_model_makes_predictions(self, clean_df):
        """Debe permitir hacer predicciones después del entrenamiento."""
        X = clean_df[['x1', 'x2']]
        y = clean_df[['y']]
        trainer = LinearRegression(X, y, train_size=0.8, split_seed=42)
        trainer.train_model()

        predictions = trainer.model.predict(X)
        assert len(predictions) == len(X)


class TestLRTrainerGetFormula:
    """Tests para get_formula() de LRTrainer."""
    
    def test_generates_formula_correctly(self, clean_df):
        """Debe generar fórmula legible correctamente."""
        X = clean_df[['x1', 'x2']]
        y = clean_df[['y']]
        trainer = LinearRegression(X, y, train_size=1.0, split_seed=42)
        trainer.train_linear_regression()
        
        formula = trainer.get_formula()
        
        assert 'y =' in formula
        assert 'x1' in formula
        assert 'x2' in formula
        assert isinstance(formula, str)
    
    def test_raises_error_if_not_trained(self, clean_df):
        """Debe lanzar error si se llama antes de entrenar."""
        X = clean_df[['x1']]
        y = clean_df[['y']]
        trainer = LinearRegression(X, y, train_size=0.8, split_seed=42)
        
        with pytest.raises(ValueError, match="Model not trained yet"):
            trainer.get_formula()