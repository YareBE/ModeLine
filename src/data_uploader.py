import pandas as pd
import sqlite3
import openpyxl
import streamlit as st
import joblib

def upload_file():
    uploaded_file = st.file_uploader(
        "Upload your dataset or a previously saved model",
                type=["csv", "xls", "xlsx", "db", "sqlite", "joblib"],
                help="Supported formats: CSV, Excel, SQLite and Joblib"
            )
    

    if uploaded_file is not None:
        extension = uploaded_file.name.split('.')[-1].lower()
        if extension != "joblib":
            with st.spinner("Loading data..."):
                df = _error_handler(uploaded_file, extension)
                if df is None:
                    return None
                elif df.empty:
                    st.error("ERROR: empty dataset, choose another one")
                    return None
                else:
                    if not df.equals(st.session_state.df):
                        from interface import reset_downstream_selections
                        reset_downstream_selections(1)
                        st.session_state.df = df
                        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        else:
            with st.spinner("Loading data..."):
                try:
                    reset_downstream_selections(1)
                    st.session_state.df = None
                    st.session_state.model = joblib.load(uploaded_file)
                    st.success(f"✅ Model '{uploaded_file.name}' correctly uploaded.")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return

@st.cache_data(show_spinner=False)
def _error_handler(file, extension):
    #Set the pointer of the file-like variable to the beginning
    file.seek(0) 
    conn = None

    try:
        if extension == 'csv':
            data = _upload_csv(file)
        elif extension in ('xls', 'xlsx'):
            data = _upload_excel(file)
        elif extension in ('db', 'sqlite'):
            conn = sqlite3.connect(':memory:') 
            data = _upload_sql(file, conn)
        return data
    
    except Exception as err:
        st.error(f"Error while reading the data: {err}")
        return

    finally:
        if extension in ('db', 'sqlite'):
            conn.close()

def _upload_csv(file):
    data = pd.read_csv(file)
    data.columns = data.columns.map(str) #To avoid invalid formats
    return data
    
def _upload_excel(file):
    data = pd.read_excel(file, engine = 'openpyxl')
    data.columns = data.columns.map(str) #To avoid invalid formats
    return data
    
def _upload_sql(file, conn):
    conn.deserialize(file.read()) #Instead of a temp file
    table = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1", conn).iloc[0, 0]
    data = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    data.columns = data.columns.map(str) #To avoid invalid formats
    return data
