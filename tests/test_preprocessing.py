import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from src.backend.preprocessing import (apply_na_handling, get_numeric_columns,
    get_na_info)



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



class TestGetNaInfo:
    """Tests para get_na_info()."""
    
    def test_identifies_columns_with_na(self, sample_df):
        """Debe identificar columnas con valores NA."""
        result = get_na_info(sample_df)
        assert set(result) == {'age', 'salary'}
    
    def test_no_missing_values(self, clean_df):
        """Debe retornar lista vacía si no hay NA."""
        result = get_na_info(clean_df)
        assert result == []
    
    def test_all_columns_with_na(self):
        """Debe detectar cuando todas las columnas tienen NA."""
        df = pd.DataFrame({
            'a': [1, np.nan],
            'b': [np.nan, 2],
            'c': [np.nan, np.nan]
        })
        result = get_na_info(df)
        assert len(result) == 3


class TestApplyNaHandling:
    """Tests para apply_na_handling()."""
    
    def test_delete_rows_method(self, sample_df):
        """Debe eliminar filas con NA."""
        result = apply_na_handling(sample_df, "Delete rows")
        assert len(result) == 3
        assert not result.isna().any().any()
    
    def test_mean_method(self, sample_df):
        """Debe rellenar con media en columnas numéricas."""
        result = apply_na_handling(sample_df, "Mean")
        assert not result[['age', 'salary']].isna().any().any()
        # Verificar que la media es correcta
        expected_age_mean = sample_df['age'].mean()
        assert result['age'].iloc[2] == expected_age_mean
    
    def test_median_method(self, sample_df):
        """Debe rellenar con mediana en columnas numéricas."""
        result = apply_na_handling(sample_df, "Median")
        assert not result[['age', 'salary']].isna().any().any()
        expected_age_median = sample_df['age'].median()
        assert result['age'].iloc[2] == expected_age_median
    
    def test_constant_method_valid(self, sample_df):
        """Debe rellenar con constante válida."""
        result = apply_na_handling(sample_df, "Constant", constant_value=999)
        assert result['age'].iloc[2] == 999.0
        assert result['salary'].iloc[3] == 999.0
    
    def test_constant_method_invalid_value(self, sample_df):
        """Debe lanzar ValueError con valor no numérico."""
        with pytest.raises(ValueError, match="Constant value must be numeric"):
            apply_na_handling(sample_df, "Constant", constant_value="invalid")
    
    def test_unrecognized_method(self, sample_df):
        """Debe retornar DataFrame sin cambios con método no reconocido."""
        result = apply_na_handling(sample_df, "UnknownMethod")
        pd.testing.assert_frame_equal(result, sample_df)
    
    def test_does_not_modify_original(self, sample_df):
        """Debe crear copia sin modificar el original."""
        original_copy = sample_df.copy()
        apply_na_handling(sample_df, "Delete rows")
        pd.testing.assert_frame_equal(sample_df, original_copy)