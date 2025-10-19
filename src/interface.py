import streamlit as st
from data_manager import DataUploader
import pandas as pd

class Interface():
    def __init__(self):
        pass
    
    def restart_app(self):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Clear cached data
        st.cache_data.clear()
        # Rerun the app
        st.rerun()

    def data_display(self, data: pd.DataFrame, type="preview" or "full"):
        if data is not None:
            if type == 'preview':
                st.dataframe(data.head(min(10, len(data))))
            else:
                st.dataframe(data)

    #cache_data optimizes computation time by storing function results
    @staticmethod  # Removes the self requirement
    @st.cache_data 
    def file_filter(data_file):
        try:
            selected_file = DataUploader(data_file)
            dataset = selected_file.error_handler()
            return dataset

        except Exception as err:
            st.error(f"Error while reading the file: {err}")
            return "error"
        
    def upload_file(self):
        file = st.file_uploader(
            "You can select only 1 file",
            type=["csv", "xls", "xlsx", "db", "sqlite"],
            key = "uploaded_file"
        )
        
        if file is None:
            st.badge("You have not selected any file yet",
                    icon=":material/warning:", color="yellow")
            return
            
        with st.status(label="Loading data", state="running", expanded = True) as status:
            data_file = self.file_filter(file)
            # If there is an error or the file is empty
            if type(data_file) == str or len(data_file) == 0:
                if type(data_file) == str:
                    st.error("Error processing file")
                else:
                    st.error("Error: Empty dataset. Select a new file")
                return
            
            status.update(label="Data correctly processed",
                         state="complete", 
                         expanded=True)
            if st.checkbox("PREVIEW"):
                self.data_display(data_file, "preview")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("CONFIRM", type='primary'):
                    # Save dataset and change pages
                    st.session_state.full_dataset = data_file
                    st.session_state.page = "DatasetManager"
                    st.rerun()
            with col3:
                if st.button("Change Dataset", type='primary'):
                    self.restart_app()

    def data_selection(self):
        if "full_dataset" in st.session_state:
            st.header("Select the features for training the model")
            self.data_display(st.session_state.full_dataset, type="full")
            if st.button("← Back to Upload"): 
                st.session_state.page = "FileUploader"
                st.rerun()
        else:
            st.error("No dataset found. Please upload a file first.")
            if st.button("Go to Upload", type = "primary"):
                st.session_state.page = "FileUploader"
                st.rerun()


    def main(self):
        st.title("ModeLine")
        #Start the page flow
        if "page" not in st.session_state:
            st.session_state.page = "FileUploader"
        # Routing basado en la página actual
        if st.session_state.page == "FileUploader":
            self.upload_file()
        elif st.session_state.page == "DatasetManager":
            self.data_selection()

if __name__ == '__main__':
    interface = Interface()
    interface.main()
