import streamlit as st
import io
import joblib


def store_model():
    """Download trained model with metadata to joblib format."""
    st.subheader("📦 Download Your Model")

    # Create a two column display for input fields
    col1, col2 = st.columns([2, 1])
    
    # Column 1: Description text area
    with col1:
        # Optional description to help users remember purpose and context
        description = st.text_area(
            "Description (optional)",
            placeholder='Example: "Model for predicting body weight"',
            height=80
        )
    
    # Column 2: Filename input
    with col2:
        # Filename input with default value
        filename = st.text_input(
            "File name",
            value="exported_model"
        )

    # Validate filename and prepare download
    if not filename or filename.strip() == '':
        st.error("❌ File name cannot be empty!")
        return

    try:
        # Build model packet dictionary with all relevant data
        # This allows the model to be loaded and used independently
        packet = {
            "model": st.session_state.model,        # Trained model object
            "description": description,             # User-provided description
            "features": st.session_state.features,  # List of feature column names
            "target": st.session_state.target,      # Target variable names
            "formula": st.session_state.formula,    # Model formula string
            "metrics": st.session_state.metrics     # Dict with performance metrics
        }
        
        # Use in-memory buffer to avoid writing to disk
        buffer = io.BytesIO()
        joblib.dump(packet, buffer)
        # Reset buffer position to beginning for reading
        buffer.seek(0)
        
        # Create download button with serialized data
        if st.download_button(
            label="⬇️ DOWNLOAD MODEL",
            data=buffer.getvalue(),           # Get bytes from buffer
            file_name=f"{filename}.joblib",   # Append .joblib extension
            mime="application/octet-stream",  # Binary file type
            type="primary",                   # Primary button styling
            use_container_width=True          # Full width button
        ):
            st.success(f"✅ Model {filename} downloaded correctly!")

    except Exception as e:
        # Catch any errors during serialization or download preparation
        st.error(f"❌ Error: {str(e)}")
