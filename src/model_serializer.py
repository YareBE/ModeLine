import streamlit as st
import io
import joblib


def store_model():
    """Render UI to download the trained model and its metadata as joblib.

    The function displays input controls for an optional description and a
    filename. It packages the model object and metadata stored in
    ``st.session_state`` into a single dictionary and serializes it to a
    Joblib binary which is available via a Streamlit download button.

    Side effects:
                - Reads ``st.session_state`` keys such as ``model``, ``features``,
                    ``target``, ``formula`` and ``metrics``.
        - When the user clicks the download button, returns the serialized
          bytes as a downloadable file.

    Returns:
        None

    Raises:
        Exception: Any error during serialization or download preparation
            is caught and displayed to the user via **st.error**.
    """
    st.subheader("📦 Download Your Model")

    # Create a two column display for input fields
    col1, col2 = st.columns([2, 1])
    
    # Column 1: Description text area
    with col1:
        # Optional description to help users remember purpose and context
        st.text_area(
            "Description (optional but recommended)",
            placeholder='Example: "Model for predicting body weight"',
            height=80, 
            key = "description"
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
        buffer = _packet_creation(st.session_state.model, st.session_state.description, st.session_state.features, 
                    st.session_state.target, st.session_state.formula, st.session_state.metrics)
        # Create download button with serialized data
        if st.download_button(
            label="⬇️ DOWNLOAD MODEL",
            data=buffer.getvalue(),           # Get bytes from buffer
            file_name=f"{filename}.joblib",   # Append .joblib extension
            mime="application/octet-stream",  # Binary file type
            type="primary",                 
            use_container_width=True        
        ):
            st.success(f"✅ Model {filename} downloaded correctly!")

    except Exception as e:
        # Catch any errors during serialization or download preparation
        st.error(f"❌ Error: {str(e)}")

def _packet_creation(model, description, features, target, formula, metrics): ##
    # Build model packet dictionary with all relevant data
    # This allows the model to be loaded and used independently
    packet = {
            "model": model,        # Trained model object
            "description": description, # User-provided description
            "features": features,  # List of feature column names
            "target": target,      # Target variable names
            "formula": formula,    # Model formula string
            "metrics": metrics     # Dict with performance metrics
        }
        
    # Use in-memory buffer to avoid writing to disk
    buffer = io.BytesIO()
    joblib.dump(packet, buffer)
    # Reset buffer position to beginning for reading
    buffer.seek(0)
    return buffer
