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

    @st.fragment
    def upload_file(self):
            st.subheader(body="_You can only select 1 file_", anchor=False)
            # Guardamos el archivo guardado en el estado
            st.session_state["file"] = st.file_uploader(
                                label="Invisible label",
                                label_visibility="collapsed",
                                type=["csv", "xls", "xlsx", "db", "sqlite"],
                                key = "uploaded_file"
                            )

            file = st.session_state["file"]  # Save file in a local variable 
            st.divider()
            if file is None:
                st.badge("You have not selected any file yet",
                        icon=":material/warning:", color="yellow")
                return
                
            with st.status(label="Loading data", state="running") as status:
                # Save the dataframe in a session_state variable ??????
                df = self.file_filter(file)
                st.session_state['dataframe'] = df
                # If there is an error or the file is empty, CHECK THIS ERROR CATCHING
                if type(df) == str or len(df) == 0:
                    if type(df) == str:
                        st.error("Error processing file")
                    else:
                        st.error("Error: Empty dataset. Select a new file")
                    return

            # End the loading widget and show a success message
            status.update(label="Data correctly processed", state="complete") 
            self.manage_dataset()

    def manage_dataset(self):
        # Dividimos la pantalla en tres columnas
        col1, col2, col3 = st.columns(spec=3, gap="large")

        # Show the three buttons in a row
        with col1: 
            # Display a menu of buttons for interacting with data
            preview = st.checkbox("PREVIEW")
        with col2:
            if st.button("CONFIRM", type='primary'):
                st.session_state["confirmed"] = True
                st.rerun()
        with col3:
            if st.button("Change Dataset", type='primary'):
                self.restart_app()  # Revisar esta forma de cambiar el dataset !!!!

        if preview:
            st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
            self.data_display(st.session_state['dataframe'], "preview")


    @st.fragment
    def features_select(self):
        # Guardamos el dataframe en una variable local
        df = st.session_state['dataframe']

        st.header("PARAMETERS SELECTION")
        # SELECT FEATURES
        st.subheader("Select the training features")
        st.session_state.selected_features = []
        for column in df.columns:
            if st.checkbox(column):
                # CONSULTAR ST.PILLS()
                st.session_state.selected_features.append(column)
        
        st.divider()
        if st.checkbox("👁️ Visualize selected dataset"):
            if len(st.session_state["selected_features"]) == 0:
                st.error("⚠️ The dataset is empty!" 
                         " Please upload or load data before proceeding.")
            else:
                # Show the selected dataset
                st.dataframe(df[st.session_state["selected_features"]])
        
        # Add space between **checkboxes** and **continue** button 
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
        
        if len(st.session_state["selected_features"]) == 0:
            st.badge("You must select at least 1 feature!",
            icon=":material/warning:", color="yellow")
        else:
            if st.button("Continue to the target selection"):
                st.rerun()

    @st.fragment
    def target_select(self):
            if "dataframe" in st.session_state:
                df = st.session_state["dataframe"]
            else:
                st.error("No dataset found. Please upload a file first.")


            # SELECT TARGET
            st.write("Selected features for now: ")
            # Show the **selected_features**
            for i in st.session_state.selected_features:
                st.write(":material/check:", i)

            st.divider()

            st.subheader("Select the target feature")
            # HABRÍA ALGUNA OTRA FORMA DE HACER ESTO 
            columns = [column for column in df.columns 
                       if column not in st.session_state["selected_features"]] 
            
            target = st.radio(label="Select a target", options=columns)
            if target:
                st.session_state["selected_target"] = target  

            st.badge("You must select a target!",
            icon=":material/warning:", color="yellow")

            
            if st.button("CONTINUE", type = "primary"):
                # Show the selected target
                st.dataframe(df[st.session_state["selected_target"]])
                # Save in session_state variable features and target
                st.session_state["features"] = df[st.session_state["selected_features"]]
                st.session_state["target"] = df[st.session_state["selected_target"]]
                st.rerun()
                        
            

if __name__ == '__main__':
    interface = Interface()
    
    st.header(body="**ModeLine**", divider="gray", 
              anchor=False)

    # Introduce space between two lines
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    if "file" not in st.session_state:
        interface.upload_file()
    else:
        if "confirmed" not in st.session_state:
            interface.manage_dataset()
        else:
            if "selected_features" not in st.session_state:
                interface.features_select()
            else: 
                interface.target_select()



    
    


    # QUÉ ES MEJOR UTILIZAR VARIABLES LOCALES (DATA_FILE) O APROVECHAR EL SESSION_STATE['DATA_FILE'] DE STREAMLIT

    
