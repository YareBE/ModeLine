#DUDAS PARA EL LUNES: 
#-Es obligatorio hacer pull requests o con clonar y hacer push llega
#-Guardado de rutas
#- Qué cantidad de datos hay que mostrar

import pandas as pd #Deberiamos comentar que pandas es una buena herramienta?
import sqlite3 
import openpyxl

path = "housing.xlsx"
extension = path.split('.')[-1]


if extension == 'csv':
    data = pd.read_csv(path)
elif extension in ('xls, xlsx'):
    data = pd.read_excel(path, engine = 'openpyxl')
elif extension in ('db', 'sqlite'):
    connection = sqlite3.connect(path)
    tabla = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", connection)["name"][0]
    data = pd.read_sql_query(f"SELECT * FROM {tabla}", connection)
else:
    pass
    ##AGREGAR ERROR DE EXTENSION, QUIZÁS CON RAISE

print(data.head())
