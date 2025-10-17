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
def file_filter(data_file):
    with st.status(label = "Loading data...", state = "running") as status:
            try:
                selected_file = DataUploader(data_file)
                dataset = selected_file.error_handler()
                status.update(label = "Data correctly uploaded",
                              state="complete")
            except Exception as err:
                status.update(label = f"Error while reading the file: {err}",
                    state = "error")
                if st.button("RETRY", type = 'primary'):
                    st.rerun()
    return dataset
    
def upload_file():
    # We select a file and save this file in the key 'file 1'
    data_file = st.file_uploader(
        "You can select only 1 file",
        type=["csv", "xls", "xlsx", "db", "sqlite"],
        key="file 1"
    )

    if data_file is None:
        st.badge("You have not selected any file yet",\
                  icon=":material/warning:", color="yellow",
                 width="stretch") # Width not working!!!
    else:
        data_file = file_filter(data_file)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            preview = st.checkbox("PREVIEW")
        with col2:
            change_dataset = st.button("CHANGE DATASET", type = "primary")
        with col3:
            confirm = st.button("CONFIRM", type = 'primary')

        if preview:
            # data_display will show the first rows of the dataset
            data_display(data_file, "preview")
        elif change_dataset:
            # We select a new file and save in 'file 2'
            data_file = st.file_uploader(
                    "You can select only 1 file",
                    type=["csv", "xls", "xlsx", "db", "sqlite"],
                    key="file 2"
                    ) # This needs to be refactor so that we can change multiple times of tile, not just one
        elif confirm:
            # data_display will show the complete dataset
            data_display(data_file, "full")
            st.badge("Dataset correctly processed", icon = ":material/check:",
                     color = "green")

def main():
    st.title("ModeLine")
    # st.set_page_config(layout="wide") # Set streamlit width to the entire page
    full_dataset = upload_file()


if __name__ == '__main__':
    main()
