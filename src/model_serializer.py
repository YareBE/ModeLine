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
    """Create a joblib-serializable packet with model and metadata (backend).
    
    Pure backend function with no Streamlit dependencies - suitable for
    unit testing.
    
    Args:
        model: Trained scikit-learn LinearRegression model.
        description (str): User-provided description (can be empty).
        features (list): List of feature column names.
        target (list): List with target variable name.
        formula (str): Model formula string representation.
        metrics (dict): Dict with performance metrics (train/test).
    
    Returns:
        io.BytesIO: In-memory buffer containing joblib-serialized packet.
    
    Raises:
        ValueError: If model is None, features/target empty, or invalid types.
        TypeError: If required parameters have wrong types.
        RuntimeError: If serialization fails.
    """
    # Validate model
    if model is None:
        raise ValueError("Model cannot be None")
    
    # Validate description
    if description is None:
        description = ""
    elif not isinstance(description, str):
        raise TypeError(f"description must be str, got {type(description).__name__}")
    
    # Validate features
    if features is None or (isinstance(features, (list, tuple)) and len(features) == 0):
        raise ValueError("features cannot be None or empty")
    if not isinstance(features, (list, tuple)):
        raise TypeError(f"features must be list/tuple, got {type(features).__name__}")
    if not all(isinstance(f, str) for f in features):
        raise TypeError("All features must be strings")
    
    # Validate target
    if target is None or (isinstance(target, (list, tuple)) and len(target) == 0):
        raise ValueError("target cannot be None or empty")
    if not isinstance(target, (list, tuple)):
        raise TypeError(f"target must be list/tuple, got {type(target).__name__}")
    if len(target) != 1:
        raise ValueError(f"target must contain exactly 1 element, got {len(target)}")
    if not isinstance(target[0], str):
        raise TypeError("target element must be string")
    
    # Validate formula
    if formula is None:
        raise ValueError("formula cannot be None")
    if not isinstance(formula, str):
        raise TypeError(f"formula must be str, got {type(formula).__name__}")
    if len(formula.strip()) == 0:
        raise ValueError("formula cannot be empty string")
    
    # Validate metrics
    if metrics is None:
        raise ValueError("metrics cannot be None")
    if not isinstance(metrics, dict):
        raise TypeError(f"metrics must be dict, got {type(metrics).__name__}")
    if len(metrics) == 0:
        raise ValueError("metrics cannot be empty")
    
    # Build packet dictionary
    try:
        packet = {
            "model": model,
            "description": description,
            "features": list(features),
            "target": list(target),
            "formula": formula,
            "metrics": metrics,
            "app": "ModeLine"
        }
    except Exception as e:
        raise RuntimeError(f"Error building packet dictionary: {str(e)}")
    
    # Serialize to in-memory buffer
    try:
        buffer = io.BytesIO()
        joblib.dump(packet, buffer)
        buffer.seek(0)
        
        # Validate buffer is not empty
        buffer_size = buffer.getbuffer().nbytes
        if buffer_size == 0:
            raise RuntimeError("Serialization produced empty buffer")
        
        return buffer
    except TypeError as e:
        raise TypeError(f"Cannot serialize packet (non-serializable object): {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error serializing packet to joblib: {str(e)}")
