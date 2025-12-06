# First of all, we need to access the backend directory from the root
import numpy as np
import streamlit as st

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from backend.model_trainer import (
    modeline_prediction,
    split_dataset,
    train_linear_regression,
    generate_formula,
    evaluate_model
)  # noqa: E402


def train_model():
    """Train a Linear Regression model using the current processed data.

    This method extracts **X** and **y** from
    **st.session_state.processed_data** using the user-selected column
    names, configures the train/test split based on session settings and
    uses **LRTrainer** to perform splitting, fitting and evaluation.

    Side effects:
        Writes trained **model**, evaluation **metrics**, predictions
        (**y_train_pred**, **y_test_pred**), readable **formula** and
        train/test subsets into **st.session_state**.

    Returns:
        None
    """
    with st.spinner("Training model..."):
        # Extract features and target from processed data
        X = st.session_state.processed_data[st.session_state.features]
        y = st.session_state.processed_data[st.session_state.target]

        # Train model
        if not st.session_state.trainset_only:
            # Normal split: use user-configured train percentage and seed
            train_ratio = st.session_state.train_size / 100
            seed = st.session_state.seed
        else:
            # Small dataset: use all data for training
            train_ratio = 1
            seed = 0
        try:
            # Store train/test subsets in session state
            (st.session_state.X_train, st.session_state.X_test,
             st.session_state.y_train, st.session_state.y_test) = (
                split_dataset(X, y, train_ratio, seed)

            )
        except Exception as e:
            st.error(f"Error: {str(e)}")

        # Train model and store in session state
        st.session_state.model = train_linear_regression(
            st.session_state.X_train, st.session_state.y_train)

        # Evaluate model and store metrics and predictions
        (st.session_state.y_train_pred,
         st.session_state.y_test_pred,
         st.session_state.metrics,
         ) = evaluate_model(st.session_state.model,
                            st.session_state.X_train,
                            st.session_state.y_train,
                            st.session_state.X_test,
                            st.session_state.y_test)

        # Generate and store a readable formula
        st.session_state.formula = generate_formula(
            st.session_state.model,
            st.session_state.features,
            st.session_state.target[0])

        st.balloons()


def predict():
    """Render a prediction UI and compute a single prediction."""
    st.subheader("Predict with ModeLine")
    st.markdown("##### Enter the input values for your prediction")

    features = (
        st.session_state.features
        or st.session_state.loaded_packet.get("features")
    )
    target = (
        st.session_state.target
        or st.session_state.loaded_packet.get("target")
    )

    if not features or not target:
        st.error("Features or target not found in session state")
        return

    model = st.session_state.get(
        "model") or st.session_state.loaded_packet.get("model")
    if model is None:
        st.error("No model found. Train or load a model first")
        return

    columns = {}
    for i in range(len(features)):
        if i % 4 == 0:
            columns[str(i // 4)] = st.columns(np.ones(4))
    inputs = np.ones(len(features))

    for name in features:
        index = features.index(name)
        with columns[str(index // 4)][index % 4]:
            inputs[index] = st.number_input(name, value=1.0, step=0.1)

    if st.button("PREDICT", type="primary"):
        try:
            prediction = modeline_prediction(model, inputs)
            st.markdown(f"#### Predicted {target[0]}: {prediction:,.3f}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
