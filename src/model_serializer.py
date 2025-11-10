import streamlit as st
import io
import joblib

def store_model(model, features, target, formula, metrics):
    
    st.subheader("Liked the performance? Save your model")
    description = st.text_input("Add an outline to be"
        " stored with your model",
        placeholder = 'Example: "Model for predicting body weight '
        ' based on height and age"')
    if description == '':
        st.warning("Adding a description is optional but recommended\n"
        " in order to avoid future confusions")

    joblib_packet = {
    "model": model, 
    "description": description,
    "features" : features,
    "target" : target, 
    "formula" : formula,
    "metrics" : metrics
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
            st.download_button(
                    label="DOWNLOAD MODEL",
                    data=buffer,
                    file_name=f"{model_name}.joblib",
                    mime="application/octet-stream", 
                    type = "primary")