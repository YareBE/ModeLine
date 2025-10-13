import pandas as pd
import sqlite3
import openpyxl
import os  

path = "housing.xlsx"

# Verificar si el archivo existe
if not os.path.exists(path):
    raise FileNotFoundError(f"File '{path}' not encountered. Verify the path.")

# Obtener la extensión del archivo
extension = path.split('.')[-1].lower()

try:
    if extension == 'csv':
        data = pd.read_csv(path)
    elif extension in ('xls', 'xlsx'):
        data = pd.read_excel(path, engine='openpyxl')
    elif extension in ('db', 'sqlite'):
        connection = sqlite3.connect(path)
        try:
            # Obtener el nombre de la primera tabla
            tabla = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", connection)["name"][0]
            data = pd.read_sql_query(f"SELECT * FROM {tabla}", connection)
        except Exception as e:
            raise ValueError(f"Error while reading the data: {e}")
        finally:
            connection.close()
    else:
        raise ValueError(f"Extension not supported: '{extension}'")
    
    # Mostrar las primeras filas del DataFrame
    print(data.head())

except FileNotFoundError as fnf_error:
    print(f"Error: {fnf_error}")
except ValueError as val_error:
    print(f"Error: {val_error}")
except pd.errors.EmptyDataError:
    print("Error: The file is empty")
except Exception as e:
    print(f"Unexpected error: {e}")