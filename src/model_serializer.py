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
            "Description (optional but recommended)",
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
            type="primary",                 
            use_container_width=True        
        ):
            st.success(f"✅ Model {filename} downloaded correctly!")

    except Exception as e:
        # Catch any errors during serialization or download preparation
        st.error(f"❌ Error: {str(e)}")


def upload_model():
    """Display loaded saved model information and metrics."""
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.header(f"{st.session_state.model_name}")

    packet = st.session_state.loaded_packet

    # Description (optional)
    st.subheader("Description")
    if packet.get("description"):
        st.info(packet["description"])
    else:
        st.warning("No description provided")

    # Formula / code string (optional)
    if packet.get("formula"):
        st.subheader("Formula")
        st.code(packet["formula"], language="python")

    # Features and target information
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

    # Performance metrics section
    st.divider()
    st.subheader("Performance Metrics")

    if packet.get("metrics"):
        metrics = packet["metrics"]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Training Set")
            if metrics.get("train"):
                metrics_train = metrics["train"]
                st.metric("R² Score", f"{metrics_train['r2']:.4f}")
                st.metric("MSE", f"{metrics_train['mse']:.4f}")
            else:
                st.warning("No training metrics")

        with col2:
            st.markdown("#### Test Set")
            if metrics.get("test"):
                metrics_test = metrics["test"]
                st.metric("R² Score", f"{metrics_test['r2']:.4f}")
                st.metric("MSE", f"{metrics_test['mse']:.4f}")
            else:
                st.warning("No test set metrics")
    else:
        st.warning("No metrics information available")

    st.divider()
