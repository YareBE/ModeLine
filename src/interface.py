import streamlit as st
from data_manager import DataUploader

def data_preview(data):
    st.dataframe(data.head(10))


if __name__ == '__main__':
    st.title("ModeLine")

    st.session_state.data_file = st.file_uploader(
        "You can select only 1 file",
        type=["csv", "xls", "xlsx", "db", "sqlite"] 
    )

    if "data_file" not in st.session_state:
        st.badge("You have not selected any file yet", icon=":material/warning:", color="yellow")
    
    if st.session_state.data_file is not None:
        try:
            whole_file = DataUploader(st.session_state.data_file)
            st.session_state.dataset = whole_file.error_handle()
        except Exception as err:
            st.error(err)
            file_reload = st.button("RETRY", type = 'primary')
            if file_reload:
                del st.session_state['data_file']
        
        st.button("DONE", type = 'primary',\
                 on_click = data_preview(st.session_state.dataset))



