import streamlit as st
from data_manager import DataUploader
import pandas as pd
import time
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
        # Create a session_state value to manage app flow
        st.session_state["upload_file"] = True
        # Add an HTML anchor to redirect the function
        st.markdown('<a name="upload_file"></a>', unsafe_allow_html=True)
        
        st.subheader(body="_You can only select 1 file_", anchor=False)
        # Guardamos el archivo guardado en el estado
        st.session_state["file"] = st.file_uploader(
                            label="Invisible label",
                            label_visibility="collapsed",
                            type=["csv", "xls", "xlsx", "db", "sqlite"],
                            key = "uploaded_file"
                        )
        
        file = st.session_state["file"]  # Save the file in a local variable
        st.divider()
        if file is None:
            st.badge("You have not selected any file yet",
                    icon=":material/warning:", color="yellow")
            return
        
        # Add a loading bar to show the upload file progress
        with st.status(label="Loading data", state="running") as status:
            # Save the dataframe in a session_state variable ??????
            df = self.file_filter(file)
            st.session_state["dataframe"] = df
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
        # Create a session_state value to manage app flow
        st.session_state["features_select"] = True
        # Add an HTML anchor to redirect the function
        st.markdown('<a name="features_select"></a>', unsafe_allow_html=True)

        # Guardamos el dataframe en una variable local
        df = st.session_state['dataframe']

        st.header("PARAMETERS SELECTION")
        # Select features 
        st.subheader("Select the training features")
        st.session_state.selected_features = []
        # Iter the columns of the dataframe and add them to the checkbox 
        for column in df.columns:
            if st.checkbox(column):
                # CONSULTAR ST.PILLS()
                st.session_state.selected_features.append(column)
        
        st.divider()
        # Add a checkbox to visualize the dataframe
        if st.checkbox("👁️ Visualize selected dataset"):
            if len(st.session_state["selected_features"]) == 0:
                st.error("⚠️ The dataset is empty!" 
                         " Please upload or load data before proceeding.")
            else:
                # Show the selected dataset
                st.dataframe(df[st.session_state["selected_features"]])
        
        # Add space between **checkboxes** and **continue** button 
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
       
        # If there are no features selected, show an error
        if len(st.session_state["selected_features"]) == 0:
            st.badge("You must select at least 1 feature!",
            icon=":material/warning:", color="yellow")
        else:
            if st.button("Continue to the target selection"):
                st.rerun()

    @st.fragment
    def target_select(self):
        # Create a session_state value to manage app flow
        st.session_state["target_select"] = True
        # Add an HTML anchor to redirect the function
        st.markdown('<a name="target_select"></a>', unsafe_allow_html=True)
        
        # Add the dataframe to a variable
        df = st.session_state["dataframe"]

        # SELECT TARGET 
        st.write("Selected features for now: ")
        # Show the **selected_features**
        for i in st.session_state.selected_features:
            st.write(":material/check:", i)

        st.divider()
        st.subheader("Select the target feature")
        # HABRÍA ALGUNA OTRA FORMA DE HACER ESTO 
        # Add columns names to a variable
        columns = [column for column in df.columns 
                   if column not in st.session_state["selected_features"]] 
        
        # Select the variable target 
        target = st.radio(label="Select a target", options=columns)
        if target:
            st.session_state["selected_target"] = target  

        st.badge("You must select a target!",
        icon=":material/warning:", color="yellow")

        # Add a confirm button to continue to the next page
        if st.button("CONFIRM", type = "primary"):
            # Show the selected target
            st.dataframe(df[st.session_state["selected_target"]])
            # Save in session_state variable features and target
            st.session_state["features"] = df[st.session_state["selected_features"]]
            st.session_state["target"] = df[st.session_state["selected_target"]]
            st.rerun()
    
    @st.fragment
    def detect_na(self):
        # Create a session_state value to manage app flow
        st.session_state["detect_na"] = True
        # Add an HTML anchor to redirect the function
        st.markdown('<a name="detect_na"></a>', unsafe_allow_html=True) 

        # Show the dataframe
        df = st.session_state["features"]

        # Count NA values in the entire dataframe
        count_na = df.isna().sum().sum() 
        # Get columns with NA values
        columns_na = df.columns[df.isna().any()].tolist()

        # If there are missing values, call **manage_na** 
        if count_na > 0:
            st.header("Let's handle the null values")
            st.badge(label=f"There are {count_na} missing values" 
                       f" in the columns: {columns_na}", 
                       icon=':material/report:', color='yellow')

            # Highlight missing values in the dataframe
            st.dataframe(df.style.highlight_null(color='yellow'))
            self.manage_na(df)
        else:
            # If there are no missing values, show a success message
            st.header("The selected dataset has no missing values")
            st.dataframe(df)
            st.toast(body="No missing values found", 
                     icon=":material/search_check_2:",
                     duration="long")
            if st.button("CONTINUE", type="primary"): 
                st.rerun()


    def manage_na(self, df: pd.DataFrame):
            # Select box to choose options for NaN substitution
            options = ['Delete rows', 'Average', 'Median', 'Constant']
            # Display a selectbox to substitute missing values
            select = st.selectbox('Select an option to substitute '
                                  'missing values:', options)

            # Input to substitute missing values with a constant
            constant = None 
            if select == "Constant":
                constant = st.text_input(label="Insert a constant to"
                                                "substitute NaN values")
            
            # Confirm the option selected
            if st.button("CONFIRM", type="primary"):
                if select == "Delete rows":
                    df = df.dropna()
                elif select == "Average":
                    df = df.fillna(df.mean(numeric_only=True))
                elif select == "Median":
                    df = df.fillna(df.median(numeric_only=True))
                elif select == "Constant":
                    # Confirm the constant value
                    if constant:
                        try:
                            constant_value = constant
                            df = df.fillna(constant_value)
                        except ValueError:
                            st.error("Please enter a valid integer constant")
                            return None
                
                # Save the actual state of the dataframe 
                st.session_state["features"] = df
                st.rerun()
        
    def final_na(self):
        # Create a session_state value to manage app flow
        st.session_state["final_na"] = True
        # Add an HTML anchor to redirect the function
        st.markdown('<a name="final_na"></a>', unsafe_allow_html=True)
        
        st.badge(label=f"Missing values filled succesfully",
                       icon=":material/thumbs_up_double:", 
                       color="green")
        # Assign the df variable to the features dataframe
        df = st.session_state["features"]
        # If the are not null values, show the dataframe
        if  not df.isna().any().any():
                st.dataframe(df)

        if st.button("CONTINUE", type="primary"): 
                st.rerun()
    

    @st.fragment
    def set_divider(self):
        # Create a session_state value to manage app flow
        st.session_state["set_divider"] = True
        # Add an HTML anchor to redirect the function
        st.markdown('<a name="final_na"></a>', unsafe_allow_html=True)

        st.header("Training set Division")
        st.subheader("The percentage (0-100) you choose for 'Training" 
                     " percentage' will be passed, as it sounds, to the"
                     " model. The rest will be used for testing it.")
        
        # Select the **train_size** using a slider
        train_size = st.slider("Training percentage", 0, 100, 80)
        # Use scikit-learn to divide the data
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state["features"], 
            st.session_state["target"], 
            train_size = train_size*0.01,      # 20% para test
            random_state = 1     # Semilla para reproducibilidad
        )

        # If train_size is lower that 50 show an error message
        if train_size < 50:
            st.error("The number of data to split the dataset is not enugh")
        else:
            st.write("Number of training rows", len(X_train))
            st.write("Number of test rows", len(X_test))
            st.badge(label=f"Data split done correctly",
                           icon=":material/done_outline:", color="green")
  

if __name__ == '__main__':
    interface = Interface()
    st.header(body="**ModeLine**", divider="gray", 
              anchor=False)

    # Introduce space between two lines
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    if "upload_file" not in st.session_state:
        interface.upload_file()
    else:
        if "features_select" not in st.session_state:
            interface.features_select()
        else: 
            if "target_select" not in st.session_state:
                interface.target_select()
            else:
                if "detect_na" not in st.session_state:
                    interface.detect_na()
                else:
                    if "final_na" not in st.session_state:
                        interface.final_na()
                    else:
                        if "set_divider" not in st.session_state:
                            interface.set_divider()
             



    
    


    # QUÉ ES MEJOR UTILIZAR VARIABLES LOCALES (DATA_FILE) O APROVECHAR EL SESSION_STATE['DATA_FILE'] DE STREAMLIT

    
