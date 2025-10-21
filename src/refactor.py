import streamlit as st
from data_manager import DataUploader
import pandas as pd
from sklearn.model_selection import train_test_split

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
                st.dataframe(data.head(10))
            else:
                st.dataframe(data)

    # Cache_data optimizes computation time by storing function results
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
            st.subheader(body="_You can only select 1 file_", anchor=False)
            file = st.file_uploader(
                label="Invisible label",
                label_visibility="collapsed",
                type=["csv", "xls", "xlsx", "db", "sqlite"],
                key = "uploaded_file"
            )
            
            st.divider()
            if file is None:
                st.badge("You have not selected any file yet",
                        icon=":material/warning:", color="yellow")
                return
                
            with st.status(label="Loading data", state="running") as status:
                # Save the dataframe in a session_state variable ??????
                st.session_state['dataframe'] = self.file_filter(file)
                df = st.session_state['dataframe']
                # If there is an error or the file is empty
                if type(df) == str or len(df) == 0:
                    if type(df) == str:
                        st.error("Error processing file")
                    else:
                        st.error("Error: Empty dataset. Select a new file")
                    return

            # End the loading widget and show a success message
            status.update(label="Data correctly processed", state="complete") 
            
            return file
    
    def manage_dataset(self):
        col1, col2, col3 = st.columns(spec=3, gap="large")

        # Show the three buttons in a row
        with col1: 
            # Display a menu of buttons for interacting with data
            preview = st.checkbox("PREVIEW")
        with col2:
            if st.button("CONFIRM", type='primary'):
                self.data_selection()
                st.rerun()
        with col3:
            if st.button("Change Dataset", type='primary'):
                self.restart_app()  # Revisar esta forma de cambiar el dataset !!!!

        if preview:
            st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
            self.data_display(st.session_state['dataframe'], "preview")

    


if __name__ == '__main__':
    interface = Interface()

    st.header(body="**ModeLine**", divider="gray", 
              anchor=False)

    # Introduce space between two lines
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    file = interface.upload_file()
    if file:
        interface.manage_dataset()



    # QUÉ ES MEJOR UTILIZAR VARIABLES LOCALES (DATA_FILE) O APROVECHAR EL SESSION_STATE['DATA_FILE'] DE STREAMLIT

    
