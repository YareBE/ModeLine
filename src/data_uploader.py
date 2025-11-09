import pandas as pd
import sqlite3
import openpyxl


def error_handler(file):
    extension = file.name.split('.')[-1].lower()
    #Set the pointer of the file-like variable to the beginning
    assert extension in ('csv', 'xls', 'xlsx', 'db', 'sqlite'), \
    "ERROR: Invalid extension"
    file.seek(0) 
    if extension == 'csv':
        data = _upload_csv()
    elif extension in ('xls', 'xlsx'):
        data = _upload_excel()
    elif extension in ('db', 'sqlite'):
        data = _upload_sql(file)
    return data

def _upload_csv(file):
    try:
        data = pd.read_csv(file)
        data.columns = data.columns.map(str) #To avoid invalid formats
        return data
    except pd.errors.EmptyDataError:
        raise Exception("ERROR: the selected file is empty")
    
def _upload_excel(file):
    try:
        data = pd.read_excel(file, engine = 'openpyxl')
        data.columns = data.columns.map(str) #To avoid invalid formats
        return data
    except pd.errors.EmptyDataError:
        raise Exception("ERROR: the selected file is empty")

def _upload_sql(file):
    conn = sqlite3.connect(':memory:') 
    try:
        conn.deserialize(file.read()) #Instead of a temp file
        table = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1", conn).iloc[0, 0]
        data = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        data.columns = data.columns.map(str) #To avoid invalid formats
        return data
    except Exception as err:
        raise Exception("Error while reading the data:", err)
    finally:
        conn.close()

