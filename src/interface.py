import streamlit as st

st.title("ModeLine")
archivo = st.file_uploader(
    "Selecciona un dataset",
    type=["csv", "xls", "xlsx", "db", "sqlite"] 
)
