import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from unittest.mock import Mock, patch
import joblib
from data_preprocess import get_numeric_columns, get_na_info, apply_na_handling

@pytest.fixture
def sample_df():
    """DataFrame de ejemplo para pruebas."""
    return pd.DataFrame({
        'age': [25, 30, np.nan, 40, 35],
        'salary': [50000, 60000, 55000, np.nan, 70000],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'score': [85.5, 90.0, 78.5, 88.0, 92.5]
    })


@pytest.fixture
def clean_df():
    """DataFrame sin valores faltantes."""
    return pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'y': [3, 5, 7, 9, 11]
    })


@pytest.fixture
def csv_file():
    """Archivo CSV de ejemplo."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


@pytest.fixture
def excel_file():
    """Archivo Excel de ejemplo."""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    return buffer



class TestGetNumericColumns:
    """Tests para get_numeric_columns()."""
    
    def test_returns_only_numeric_columns(self, sample_df):
        """Debe retornar solo columnas numéricas."""
        result = get_numeric_columns(sample_df)
        assert set(result) == {'age', 'salary', 'score'}
        assert 'name' not in result
    
    def test_empty_dataframe(self):
        """Debe manejar DataFrame vacío."""
        df = pd.DataFrame()
        result = get_numeric_columns(df)
        assert result == []
    
    def test_no_numeric_columns(self):
        """Debe retornar lista vacía si no hay columnas numéricas."""
        df = pd.DataFrame({'text': ['a', 'b'], 'category': ['x', 'y']})
        result = get_numeric_columns(df)
        assert result == []
    
    def test_all_numeric_columns(self):
        """Debe retornar todas las columnas si todas son numéricas."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0], 'c': [5, 6]})
        result = get_numeric_columns(df)
        assert len(result) == 3