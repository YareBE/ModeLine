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


def display_saved_models(loaded_packet, model_name):
        """Display and load previously saved models"""
        # Header con badge del modelo
        col_title, col_badge = st.columns([3, 1])
        with col_title:
            st.header(f"{model_name}")

        packet = loaded_packet
        
        # Descripción en un expander para no ocupar tanto espacio
        st.subheader("Description")
        if packet.get("description"):
            st.info(packet["description"])
        else:
            st.warning("No description provided")
        
        # Fórmula
        if packet.get("formula"):
            st.subheader("Formula")
            st.code(packet["formula"], language="python")
        
        # Features y Target en una sola fila más compacta
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Features:**")
            if packet.get("features"):
                with st.container(border=True):
                    for feat in packet["features"]:
                        st.markdown(f"• `{feat}`")
            else:
                st.warning("No features information")
        
        with col2:
            st.markdown("**Target:**")
            if packet.get("target"):
                with st.container(border=True):
                    target = packet["target"][0]
                    st.markdown(f"• `{target}`")
            else:
                st.warning("No target information")
        
        # Métricas con mejor visualización
        st.divider()
        st.subheader("Performance Metrics")
        
        if packet.get("metrics"):
            metrics = packet["metrics"]
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Training Set")
                if metrics.get('train'):
                    metrics_train = metrics['train']
                    st.metric("R² Score", f"{metrics_train['r2']:.4f}")
                    st.metric("MSE", f"{metrics_train['mse']:.4f}")
                else:
                    st.warning("No training metrics available")
            
            with col2:
                st.markdown("#### Test Set")
                if metrics.get('test'):
                    metrics_test = metrics['test']
                    st.metric("R² Score", f"{metrics_test['r2']:.4f}")
                    st.metric("MSE", f"{metrics_test['mse']:.4f}")
                else:
                    st.warning("No test set metrics available")

        else:
            st.warning("No metrics information available")
        
        st.divider()
        

