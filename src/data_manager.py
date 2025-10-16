import pandas as pd
import sqlite3
import openpyxl

class DataUploader:
    """Class responsible for the correct uploading and error handling
    of the selected file in the interface
    """
    def __init__(self, file):
        self._file = file
    
    def error_handler(self):
        extension = self._file.name.split('.')[-1].lower()
        #Set the pointer of the file-like variable to the beginning
        assert extension in ('csv', 'xls', 'xlsx', 'db', 'sqlite'), \
        "ERROR: Invalid extension"
        self._file.seek(0) 
        if extension == 'csv':
            data = self._upload_csv()
        elif extension in ('xls', 'xlsx'):
            data = self._upload_excel()
        elif extension in ('db', 'sqlite'):
            data = self._upload_sql(self._file)
        return data
    
    def _upload_csv(self):
        try:
            data = pd.read_csv(self._file)
            return data
        except pd.errors.EmptyDataError:
            raise Exception("ERROR: the selected file is empty")
        
    def _upload_excel(self):
        try:
            data = pd.read_excel(self._file, engine = 'openpyxl')
            return data
        except pd.errors.EmptyDataError:
            raise Exception("ERROR: the selected file is empty")
    
    def _upload_sql(self, file):
        conn = sqlite3.connect(':memory:') 
        try:
            conn.deserialize(file.read()) #Instead of a temp file
            table = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1", conn).iloc[0, 0]
            data = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            return data
        except Exception as err:
            raise Exception("Error while reading the data:", err)
        finally:
            conn.close()
