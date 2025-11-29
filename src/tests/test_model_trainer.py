from model_trainer import _modeline_prediction, predict 
from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st
import numpy as np
import unittest 
from src.model_trainer import LinearRegression


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




    