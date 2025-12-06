"""Data visualization and display utilities for model results.

This module provides functions for rendering DataFrames with syntax highlighting,
visualizing model performance metrics, and plotting 1D/3D regression results.
All functions include type hints and comprehensive docstrings.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import List, Any
import pandas as pd


@st.cache_data(show_spinner=False)
def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of numeric column names from a DataFrame.

    Inspects a DataFrame and returns column names containing numeric data types
    (int, float, etc.). Excludes columns with object, string, or categorical types.

    Args:
        df (pd.DataFrame): Input DataFrame to inspect.

    Returns:
        List[str]: Column names of numeric dtype.
    """
    return df.select_dtypes(include=['number']).columns.tolist()


@st.cache_data(show_spinner=False)
def get_na_info(df: pd.DataFrame) -> List[str]:
    """Return a list of columns that contain missing values.

    Scans the DataFrame to identify any columns with NaN/None values
    and returns their names. Useful for deciding NA handling strategy.

    Args:
        df (pd.DataFrame): Input DataFrame to inspect.

    Returns:
        List[str]: Column names that have at least one NA/NaN value.
    """
    cols_with_na = df.columns[df.isna().any()].tolist()
    return cols_with_na


def display_dataframe() -> None:
    """Render a preview of the active DataFrame with a selectable row window.

    The function displays a slice of st.session_state.df and, if
    st.session_state.processed_data exists, substitutes the selected
    features/target columns with the processed values. It offers a
    100-row-window selector to avoid rendering very large tables.

    Side effects:
        Reads from st.session_state keys: df, processed_data,
        features, target and writes to the Streamlit UI.

    Returns:
        None: The UI is rendered directly via Streamlit.
    """
    df = st.session_state.df.copy()
    processed_data = st.session_state.processed_data

    if processed_data is not None:
        df[st.session_state.features + st.session_state.target] = (
            processed_data
        )

    st.markdown("#### Dataset Preview")

    # Row range selector - show 100-row windows to avoid rendering huge tables
    col1, _ = st.columns([3, 1])
    with col1:
        available_rows = [
            [i, min(i + 100, len(df))] for i in range(0, len(df), 100)
        ]

        # Default to the first window (index=0)
        rows_displayed = st.selectbox(
            "Choose the range of rows to display",
            options=available_rows,
            index=0,
        )

    # Small legend so users can tell which columns are features/target
    st.caption("🔵 Blue = Features | 🔴 Red = Target")

    # Slice and style the DataFrame for display
    display_df = df[rows_displayed[0]:rows_displayed[1]]
    styled_df = style_dataframe(
        df=display_df,
        features=st.session_state.features,
        target=st.session_state.target,
    )
    st.dataframe(styled_df, use_container_width=True, height=400)


def style_dataframe(
    df: pd.DataFrame,
    features: List[str],
    target: List[str]
) -> Any:
    """Return a pandas Styler that highlights feature and target columns.

    Applies subtle background colors to distinguish feature columns (blue)
    from target columns (red) for better visual clarity in DataFrames.

    Args:
        df (pd.DataFrame): DataFrame slice to style.
        features (List[str]): List of column names considered features.
        target (List[str]): List (usually length 1) of column name considered target.

    Returns:
        pandas.io.formats.style.Styler: Styled DataFrame ready for Streamlit display.
    """
    def highlight_columns(col: pd.Series) -> List[str]:
        # Apply subtle blue for features, light red for targets
        if col.name in features:
            return ["background-color: #e3f2fd"] * len(col)
        elif col.name in target:
            return ["background-color: #ffb3b3"] * len(col)
        else:
            return [""] * len(col)

    return df.style.apply(highlight_columns)


def visualize_results() -> None:
    """Render trained model performance metrics and the model formula.

    Reads the trained model and metrics from st.session_state and
    displays R² and MSE for training and, when available, test sets. If the
    app is running with a loaded model packet, metrics and formula are also
    shown via the packet contents.

    Returns:
        None: UI elements are rendered in Streamlit.
    """
    if st.session_state.model is not None:
        # Use st.markdown with HTML to personalize the size of the text
        st.markdown(
            "<p style='text-indent: 30px;'>"
            "<span style='font-size:20px; color:blue;'>"
            f"{st.session_state.formula}</span></p>",
            unsafe_allow_html=True,
        )

        # Split UI into two columns: training and test metrics
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Training Set")
            metrics_train = st.session_state.metrics["train"]
            st.metric("R² Score", f"{metrics_train['r2']:.4f}")
            st.metric("MSE", f"{metrics_train['mse']:.4f}")

        with col2:
            # If there is test data, show test metrics; otherwise warn
            if not st.session_state.trainset_only:
                st.markdown("#### Test Set")
                metrics_test = st.session_state.metrics["test"]
                st.metric("R² Score", f"{metrics_test['r2']:.4f}")
                st.metric("MSE", f"{metrics_test['mse']:.4f}")
            else:
                st.markdown("#### Warning")
                st.warning(
                    "Performance metrics only for training set. "
                    "Results may not be realistic due to small dataset."
                )


def plot_results():
    """Create interactive Plotly visualizations comparing predictions and
    actual target values.

    Depending on the number of features, the function will create:
        - A default Actual vs Predicted scatter plot.
        - A 2D regression line plot for single-feature models.
        - A 3D surface plot for two-feature models (regression plane).

    The function expects several keys to be present in **st.session_state**
    (**X_train**, **X_test**, **y_train**, **y_test**, **y_train_pred**,
    **y_test_pred**, and **model**) and will raise a **ValueError** if the
    model or predictions are missing.

    Returns:
        None: Interactive charts are rendered using Streamlit's Plotly
        integration.
    """
    # Extract session state
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    y_train_pred = st.session_state.y_train_pred
    y_test_pred = st.session_state.y_test_pred
    model = st.session_state.model

    # Basic validation to guide users if they call this too early
    if model is None:
        raise ValueError("Model not trained yet. Call train_model() first")

    if y_train_pred is None:
        raise ValueError("Model not tested yet. Call test_model() first")

    # Save the number of features
    n_features = X_train.shape[1]

    try:
        # Create default plot
        def_fig = _create_default_plot(
            y_train, y_train_pred, y_test, y_test_pred
        )
        st.plotly_chart(def_fig, use_container_width=True)
        st.divider()

        # Create specialized plots for 1 or 2 features
        if n_features == 1:
            fig = _create_1d_plot(X_train, X_test, y_train, y_test,
                                  y_train_pred, model)
            st.plotly_chart(fig, use_container_width=True)
            st.divider()
        elif n_features == 2:
            fig = _create_3d_plot(X_train, X_test, y_train, y_test, model)
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

    except Exception as e:
        raise RuntimeError(f"Error plotting results: {str(e)}")


# Aux function for creating the plots
def _make_trace(x, y, name, color, size=5, opacity=0.5, symbol=None):
    """Create a Plotly Scattergl trace for 2D scatter plots.

    Args:
        x (array-like): X values for the scatter trace.
        y (array-like): Y values for the scatter trace.
        name (str): Trace name shown in the legend.
        color (str): Marker color (hex or named color).
        size (int, optional): Marker size. Defaults to 5.
        opacity (float, optional): Marker opacity between 0 and 1.
        symbol (str, optional): Marker symbol name (e.g., 'x'). Defaults to None.

    Returns:
        plotly.graph_objects.Scattergl: Configured scatter trace.
    """
    return go.Scattergl(
        x=x, y=y, mode='markers', name=name,
        marker={
            'size': size, 'color': color, 'opacity': opacity, 'symbol': symbol
            }
    )


def _create_default_plot(y_train, y_train_pred, y_test, y_test_pred):
    """Create an Actual vs Predicted scatter plot using Plotly.

    Args:
        y_train (pandas.DataFrame): True training target values.
        y_train_pred (array-like): Predicted values for the training set.
        y_test (pandas.DataFrame or None): True test target values or None.
        y_test_pred (array-like or None): Predicted test values or None.

    Returns:
        plotly.graph_objects.Figure: Configured scatter figure.
    """
    def_fig = go.Figure()

    y_train_pred = y_train_pred.ravel()
    y_train = y_train.values.ravel()

    def_fig.add_trace(
        _make_trace(
            y_train, y_train_pred,
            name='Train', size=5, color='#2E86AB', opacity=0.5
        )
    )

    all_values = np.concatenate([y_train, y_train_pred])

    # If test predictions are available, add them and include them in range
    if not st.session_state.trainset_only:
        y_test_pred = y_test_pred.ravel()
        y_test = y_test.values.ravel()

        def_fig.add_trace(
            _make_trace(
                y_test, y_test_pred,
                name='Test', size=6, color='#A23B72', opacity=0.6,
                symbol='x'
            )
        )

        all_values = np.concatenate([all_values, y_test, y_test_pred])

    # Compute small margin to avoid points sitting on the axes
    min_val, max_val = all_values.min(), all_values.max()
    margin = (max_val - min_val) * 0.05
    min_val -= margin
    max_val += margin

    def_fig.add_trace(
        go.Scattergl(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            name='Perfect',
            line={'color':'black', 'width':2, 'dash':'dash'}         )
    )

    def_fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual',
        yaxis_title='Predicted',
        template='plotly_white',
        height=700,
        yaxis={'scaleanchor':"x", 'scaleratio':1}
    )

    return def_fig


def _create_1d_plot(X_train, X_test, y_train, y_test, y_train_pred, model):
    """Create a 2D regression plot for models with a single feature.

    Args:
        X_train (pandas.DataFrame): Training features (single column).
        X_test (pandas.DataFrame or None): Test features or None.
        y_train (pandas.DataFrame): Training target values.
        y_test (pandas.DataFrame or None): Test target values or None.
        y_train_pred (array-like): Predictions for the training set.
        model: Trained model instance implementing **predict**.

    Returns:
        plotly.graph_objects.Figure: 2D scatter+regression line figure.
    """
    x_train = X_train.iloc[:, 0].to_numpy()
    y_train_vals = y_train.iloc[:, 0].to_numpy()

    fig = go.Figure()
    fig.add_trace(_make_trace(
        x_train, y_train_vals,
        name='Train', color='#2E86AB'
    ))

    # Scatter points for test set (if present)
    if not st.session_state.trainset_only:
        fig.add_trace(_make_trace(
            x_train, y_train_vals,
            name='Test', color='#A23B72', size=6, opacity=0.7, symbol='x'
        ))

    # Regression line
    x_min, x_max = x_train.min(), x_train.max()
    x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_pred_range = model.predict(x_range)

    fig.add_trace(go.Scattergl(
        x=x_range.ravel(),
        y=y_pred_range.ravel(),
        mode='lines',
        name='Regression',
        line={'color':'#F18F01', 'width':3}
    ))

    fig.update_layout(
        title="Linear Regression: Feature vs Target",
        xaxis_title=X_train.columns[0],
        yaxis_title=y_train.columns[0],
        template='plotly_white',
        height=600
    )

    return fig


def _create_3d_plot(X_train, X_test, y_train, y_test, model):
    """Create a 3D regression visualization (two features -> target).

    The function plots training (and test, if present) points in 3D and
    overlays the regression plane predicted by the fitted model.

    Args:
        X_train (pandas.DataFrame): Training features (two columns).
        X_test (pandas.DataFrame or None): Test features or None.
        y_train (pandas.DataFrame): Training target values.
        y_test (pandas.DataFrame or None): Test target values or None.
        model: Trained model instance implementing **predict** for 2D inputs.

    Returns:
        plotly.graph_objects.Figure: 3D scatter and surface figure.
    """

    def _make_trace(x, y, z, name, color, size=3, opacity=0.7, symbol=None):
        return go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers", name=name,
            marker={'size':size, 'color':color, 'opacity':opacity, 'symbol':symbol}
        )

    x1_train = X_train.iloc[:, 0].to_numpy()
    x2_train = X_train.iloc[:, 1].to_numpy()
    z_train = y_train.iloc[:, 0].to_numpy()

    fig = go.Figure()
    fig.add_trace(_make_trace(
        x1_train, x2_train, z_train,
        name='Train', color='#2E86AB'
    ))

    # Test points in 3D (if present)
    if not st.session_state.trainset_only:
        x1_test = X_test.iloc[:, 0].to_numpy()
        x2_test = X_test.iloc[:, 1].to_numpy()
        z_test = y_test.iloc[:, 0].to_numpy()

        fig.add_trace(_make_trace(
            x1_test, x2_test, z_test,
            name='Test', size=4, color='#A23B72', opacity=0.8, symbol='x'
        ))

    # Regression plane
    x1_range = np.linspace(x1_train.min(), x1_train.max(), 25)
    x2_range = np.linspace(x2_train.min(), x2_train.max(), 25)

    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    x_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    z_grid = model.predict(x_grid).reshape(x1_grid.shape)

    fig.add_trace(go.Surface(
        x=x1_range, y=x2_range, z=z_grid,
        name='Regression Plane',
        colorscale='YlOrRd',
        opacity=0.6,
        showscale=False
    ))

    fig.update_layout(
        title="3D Linear Regression",
        scene={
            'xaxis_title':X_train.columns[0],
            'yaxis_title':X_train.columns[1],
            'zaxis_title':y_train.columns[0],
        },
        template='plotly_white',
        height=700
    )

    return fig


def display_uploaded_model():
    """Render the UI for a previously loaded model packet.

    The function reads the loaded packet from **st.session_state.loaded_packet**
    and displays description, formula, feature/target configuration and
    performance metrics in a human-friendly format.

    Side effects:
        - Reads **st.session_state.loaded_packet** and **st.session_state.model_name**.
        - Renders multiple Streamlit components (headers, metrics, code blocks,
            and lists) to present model metadata and performance.

    Returns:
        None
    """
    display_title()
    packet = st.session_state.loaded_packet
    display_description(packet)
    display_formula(packet)
    display_configuration(packet)
    display_metrics(packet) 

def display_title():
    """Display the model title"""
    col_title, _ = st.columns([3, 1])
    with col_title:
        st.header(f"{st.session_state.model_name}")

def display_description(packet):
    """Display the model description (Optional)"""
    st.subheader("Description")
    if packet.get("description"):
        st.info(packet["description"])
    else:
        st.warning("No description provided")

def display_formula(packet):
    """Display the model formula"""
    if packet.get("formula"):
        st.subheader("Formula")
        st.code(packet["formula"], language="python")

def display_configuration(packet):
    """Display features and target configuration"""
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Features:**")
        if packet.get("features"):
            with st.container(border=True):
                for feat in packet["features"]:
                    st.markdown(f"• *{feat}*")
        else:
            st.warning("No features information")

    with col2:
        st.markdown("**Target:**")
        if packet.get("target"):
            with st.container(border=True):
                target = packet["target"][0]
                st.markdown(f"• *{target}*")
        else:
            st.warning("No target information")

def display_metrics(packet):
    """Display performance metrics"""
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


def display_dataset_info(df):
    """Display basic dataset info in the Streamlit UI and return numeric columns.

    This helper renders basic metrics (rows/columns), warns about missing
    values and very small datasets, and returns the list of numeric columns
    available for modeling.

    Args:
        df (pandas.DataFrame): Dataset to inspect and display.

    Returns:
        list: Names of numeric columns found in **df**.
    """
    available_columns = get_numeric_columns(df)
    cols_with_na = get_na_info(df)

    # Show basic metrics of the dataset
    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Cols", len(df.columns))

    # Show warning of missing values
    if cols_with_na:
        # Limit the list of the first 3 columns to avoid saturating the UI
        na_list = ', '.join(cols_with_na[:3])
        suffix = ' ...' if len(cols_with_na) > 3 else ''
        st.caption(f"**Columns with NA values:** {na_list}{suffix}")

    # Warn if the dataset is too small to do a split train/test
    if len(df) < 10:
        msg = ("WARNING: Dataset too small. Training set will contain "
               "all data, resulting in empty test set.")
        st.warning(msg)
        # Flag to use 100% of data in training
        st.session_state.trainset_only = True

    st.divider()
    return available_columns
