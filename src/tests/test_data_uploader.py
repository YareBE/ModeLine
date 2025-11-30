import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from unittest.mock import Mock, patch
import joblib
from src.data_uploader import _upload_csv, _upload_excel, _error_handle


class TestUploadCsv:
    """Tests para _upload_csv()."""
    
    def test_reads_csv_successfully(self, csv_file):
        """Debe leer CSV correctamente."""
        result = _upload_csv(csv_file)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['a', 'b']
    
    def test_converts_column_names_to_string(self):
        """Debe convertir nombres de columnas a string."""
        df = pd.DataFrame({0: [1, 2], 1: [3, 4]})
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        result = _upload_csv(buffer)
        assert all(isinstance(col, str) for col in result.columns)
    
    def test_empty_csv_raises_error(self):
        """Debe lanzar ValueError con CSV vacío."""
        buffer = BytesIO(b"")
        with pytest.raises(ValueError, match="CSV file is empty"):
            _upload_csv(buffer)





class TestUploadExcel:
    """Tests para _upload_excel()."""
    
    def test_reads_excel_successfully(self, excel_file):
        """Debe leer Excel correctamente."""
        result = _upload_excel(excel_file)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['a', 'b']
    
    def test_converts_column_names_to_string(self, excel_file):
        """Debe convertir nombres de columnas a string."""
        result = _upload_excel(excel_file)
        assert all(isinstance(col, str) for col in result.columns)