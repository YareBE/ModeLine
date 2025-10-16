import streamlit as st
import time
from data_manager import DataUploader
import pandas as pd

def data_display(data: pd.DataFrame, type = "preview" or "full"):
    if type == 'preview':
        st.dataframe(data.head(min(10, len(data))))
    else:
        st.dataframe(data)

#cache_data optimizes computation time by storing function results
@st.cache_data 
def file_filter(data_file):
    with st.status(label = "Loading data...", state = "running") as status:
            try:
                selected_file = DataUploader(data_file)
                dataset = selected_file.error_handler()
            except Exception as err:
                status.update(label = f"Error while reading the file: {err}",
                    state = "error")
                if st.button("RETRY", type = 'primary'):
                    st.rerun()
            status.update(label = "blue-background[Data correctly uploaded]",
                         state = "complete")
    return dataset
    
def upload_file():
    data_file = st.file_uploader(
        "You can select only 1 file",
        type=["csv", "xls", "xlsx", "db", "sqlite"] 
    )
    if data_file is None:
        st.badge("You have not selected any file yet",\
                  icon=":material/warning:", color="yellow")
    else:
        data_file = file_filter(data_file)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.checkbox("PREVIEW"):
                data_display(data_file, "preview")
        with col2:
            if st.button("CHANGE DATASET", type = "primary"):
                st.rerun()
        with col3:
            if st.button("CONFIRM", type = 'primary'):
                dataset = data_file
        


        
def main():
    st.title("ModeLine")
    full_dataset = upload_file()
    st.badge("Dataset correctly processed", icon = ":material/check:", color = "green")
    data_display(full_dataset, type = "full")



if __name__ == '__main__':
    main()