import streamlit as st
import io
import joblib

def store_model():
    st.subheader("Liked the performance? Save your model")
    description = st.text_input("Add an outline to be"
        " stored with your model (enter to apply):",
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

    model_name = st.text_input("File name without extension (enter to apply):", value="exported_model")
    if model_name == '':
        st.warning("This field can't be empty!")
    else:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
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
