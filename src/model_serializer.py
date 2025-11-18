import streamlit as st
import io
import joblib


def store_model():
    """Download trained model with metadata to joblib format."""
    st.subheader("📦 Download Your Model")

    # Input fields
    col1, col2 = st.columns([2, 1])

    with col1:
        description = st.text_area(
            "Description (optional)",
            placeholder='Example: "Model for predicting body weight"',
            height=80
        )

    with col2:
        filename = st.text_input(
            "File name",
            value="exported_model"
        )

    # Validate and prepare download
    if not filename or filename.strip() == '':
        st.error("❌ File name cannot be empty!")
        return

    try:
        # Build and serialize model packet
        packet = {
            "model": st.session_state.model,
            "description": description,
            "features": st.session_state.features,
            "target": st.session_state.target,
            "formula": st.session_state.formula,
            "metrics": st.session_state.metrics
        }

        buffer = io.BytesIO()
        joblib.dump(packet, buffer)
        buffer.seek(0)

        if st.download_button(
            label="⬇️ DOWNLOAD MODEL",
            data=buffer.getvalue(),
            file_name=f"{filename}.joblib",
            mime="application/octet-stream",
            type="primary",
            use_container_width=True
        ):
            st.success(f"✅ Model {filename} downloaded correctly!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
