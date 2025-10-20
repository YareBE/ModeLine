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
                st.dataframe(data.head(min(10, len(data))))
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
                    st.session_state.page = "NaHandler"
                    st.rerun()
            with col3:
                if st.button("Change Dataset", type='primary'):
                    self.restart_app()

    def data_selection(self):
        if "full_dataset" in st.session_state:
            st.header("PARAMETERS SELECTION")
            df = st.session_state.full_dataset
            
            if st.button("← Back to NA handler"): 
                st.session_state.page = "NaHandler"
                st.rerun()

            if st.session_state.subpage == "features":
                if "selected_target" in st.session_state:
                    st.write("Actual selected target", st.session_state.selected_target)
                st.subheader("Select the training features")
                st.session_state.selected_features = []
                for column in df.columns:
                    if st.checkbox(column):
                        st.session_state.selected_features.append(column)
                st.dataframe(df[st.session_state.selected_features])
                
                if len(st.session_state.selected_features) == 0:
                    st.badge("You must select at least 1 feature!",
                    icon=":material/warning:", color="yellow")
                else:
                    if st.button("Continue to the target selection"):
                        st.session_state.subpage = "target"
                        st.rerun()

            elif st.session_state.subpage == "target":
                st.write("Selected features for now: ")
                for i in st.session_state.selected_features:
                    st.write(":material/check:", i)

                st.divider()

                st.session_state.selected_target = None
                st.subheader("Select the target feature")
                for column in df.columns:
                    if column not in st.session_state.selected_features:
                        if st.checkbox(column):
                            st.session_state.selected_target = column
                if st.session_state.selected_target is not None:
                    st.dataframe(df[st.session_state.selected_target])
                
                col1, col2 = st.columns([1,1])
                with col1:
                    if st.button("Back to features selection"):
                        st.session_state.subpage = "features"
                        st.rerun()
                with col2:
                    if st.session_state.selected_target is None:
                        st.badge("You must select a target!",
                        icon=":material/warning:", color="yellow")
                    else:
                        if st.button("CONTINUE", type = "primary"):
                            st.session_state["features"] = df[st.session_state.selected_features]
                            st.session_state["target"] = df[st.session_state.selected_target]
                            st.session_state.page = "SetDivider"
                            st.rerun()
                
            
        else:
            st.error("No dataset found. Please upload a file first.")
            if st.button("Go to Upload", type = "primary"):
                st.session_state.page = "FileUploader"
                st.rerun()
    
    def detect_na(self):
        st.header("Let's handle the null values")
        df = st.session_state.full_dataset
        # Count NA values in the entire dataframe
        count_na = df.isna().sum().sum() 
        # Get columns with NA values
        columns_na = df.columns[df.isna().any() > 0].tolist()

        if st.button("← Back to Upload"): 
                st.session_state.page = "FileUploader"
                st.rerun()

        if count_na > 0:
            st.badge(label=f"There are {count_na} missing values" 
                       f" in the columns: {columns_na}", 
                       icon=':material/report:', color='yellow')
            # Highlight missing values in the dataframe
            st.dataframe(df.style.highlight_null(color='yellow'))
            
            # Select box to choose options for NaN substitution
            options = ['Delete rows', 'Average', 'Median', 'Constant']
            select = st.selectbox('Select an option to substitute '
                                  'missing values:', options)

            constant = None # Text input for constant
            if select == "Constant":
                constant = st.text_input(label="Insert a constant to substitute NaN values")

            if st.button("CONFIRM", type="primary"):
                if select == "Delete rows":
                    df = df.dropna()
                elif select == "Average":
                    df = df.fillna(df.mean(numeric_only=True).astype(int))
                elif select == "Median":
                    df = df.fillna(df.median(numeric_only=True).astype(int))
                elif select == "Constant":
                    if constant:
                        try:
                            constant_value = int(constant)
                            df = df.fillna(constant_value)
                        except ValueError:
                            st.error("Please enter a valid integer constant")
                            return None
                st.session_state.full_dataset = df
                st.badge(label=f"Missing values  "
                                        f"filled succesfully with: {select}",
                                        icon=":material/thumbs_up_double:", color="green")
                
        if not df.isna().any().any():
            if st.button("CONTINUE TO FEATURES SELECTION", type = "primary"):
                st.session_state.full_dataset = df
                st.session_state.page = "DatasetManager"
                st.session_state.subpage = "features"  # Default subpage
                st.rerun()

    def set_divider(self):
        st.header("Training set Division")
        st.subheader("The percentage (0-100) you choose for 'Training percentage'" \
        "will be passed, as it sounds, to the model. The rest will be used for testing it.")
        
        train_size = st.slider("Training percentage", 0, 100, 80)
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state['features'], 
            st.session_state['target'], 
            train_size = train_size*0.01,      # 20% para test
            random_state = 1     # Semilla para reproducibilidad
        )
        st.write("Number of training rows", len(X_train))
        st.write("Number of test rows", len(X_test))

    

        

        
    def main(self):
        st.title("ModeLine")
        st.divider()
        #Start the page flow
        if "page" not in st.session_state:
            st.session_state.page = "FileUploader"
        
        if st.session_state.page == "FileUploader":
            self.upload_file()
        elif st.session_state.page == "NaHandler":
            self.detect_na()
        elif st.session_state.page == "DatasetManager":
            self.data_selection()
        elif st.session_state.page == "SetDivider":
            self.set_divider()

if __name__ == '__main__':
    interface = Interface()
    interface.main()
