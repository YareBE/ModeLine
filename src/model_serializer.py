import streamlit as st
import io
import joblib

def store_model():
    st.subheader("Liked the performance? Save your model")
    description = st.text_input("Add an outline to be"
        " stored with your model",
        placeholder = 'Example: "Model for predicting body weight '
        ' based on height and age"')
    if description == '':
        st.warning("Adding a description is optional but recommended\n"
        " in order to avoid future confusions")

    joblib_packet = {
    "model": st.session_state.model, 
    "description": description,
    "features" : st.session_state.features,
    "target" : st.session_state.target, 
    "formula" : st.session_state.formula,
    "metrics" : st.session_state.metrics
    }

    model_name = st.text_input("File name (without extension):", value="exported_model")
    if model_name == '':
        st.warning("This field can't be empty!")
    else:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            buffer = io.BytesIO()
            joblib.dump(joblib_packet, buffer)
            buffer.seek(0) 
            if st.download_button(
                    label="DOWNLOAD MODEL",
                    data=buffer,
                    file_name=f"{model_name}.joblib",
                    mime="application/octet-stream", 
                    type = "primary"):
                
                    st.success("✅ The model has been saved successfully!")
        
                    st.subheader("What would you like to do next?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Load Saved Models", use_container_width=True, type="secondary"):
                            st.session_state.show_saved_models = True
                            st.rerun()
                    
                    with col2:
                        if st.button("Create a New Model", use_container_width=True, type="secondary"):
                            # Limpiar todo el session state
                            for key in list(st.session_state.keys()):
                                if key != "file_uploader_key":
                                    del st.session_state[key]
                            st.session_state.file_uploader_key += 1
                            st.rerun()
