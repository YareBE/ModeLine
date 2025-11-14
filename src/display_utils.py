import streamlit as st
import numpy as np
import plotly.graph_objects as go

def display_dataframe():
        """Fragment for dataframe display to avoid full rerun on checkbox toggle"""
        df = st.session_state.df.copy()
        processed_data = st.session_state.processed_data
        if processed_data is not None:
            df[st.session_state.features +\
                st.session_state.target] = processed_data
        
        st.markdown("#### Dataset Preview")
        
        # Add controls for data display
        col1, col2 = st.columns([3, 1])
        with col1:
            available_rows = [[i, i + 100 if i+100 <= len(df) else len(df)] \
                            for i in range(0, len(df), 100)]
            rows_displayed = st.selectbox("Choose the range of rows to be" \
            " displayed", options = available_rows, index = 0 if 
            "rows_displayed" not in st.session_state else\
                    st.session_state.rows_displayed[0]//100)

            st.session_state.rows_displayed = rows_displayed
        
        # Display dataframe with limited rows for performance
        st.caption("🔵 Blue = Features | 🔴 Red = Target")
        
        # Only style and show limited rows
        display_df = df[rows_displayed[0]:rows_displayed[1]]
        styled_df = style_dataframe(
            _df = display_df,
            features = st.session_state.features,
            target = st.session_state.target,
            axis = 0
        )
        st.dataframe(styled_df, use_container_width=True, height=400)


def style_dataframe( _df, features, target, *args, **kwargs):
        """Cache styled dataframe to avoid recalculation"""
        def highlight_columns(col, *args, **kwargs):
            if col.name in features:
                return ['background-color: #e3f2fd'] * len(col)
            elif col.name in target:
                return ['background-color: #ffb3b3'] * len(col)
            else:
                return [''] * len(col)
        
        return _df.style.apply(highlight_columns)
     

def visualize_results():
        """Display model results"""
        if st.session_state.model is not None:
            st.info(st.session_state.formula)
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Training Set")
                metrics_train = st.session_state.metrics['train']
                st.metric("R² Score", f"{metrics_train['r2']:.4f}")
                st.metric("MSE", f"{metrics_train['mse']:.4f}")
            with col2:
                if not st.session_state.trainset_only:
                        st.markdown("#### Test Set")
                        metrics_test = st.session_state.metrics['test']
                        st.metric("R² Score", f"{metrics_test['r2']:.4f}")
                        st.metric("MSE", f"{metrics_test['mse']:.4f}")
                else:
                    st.subheader("\tIMPORTANT")
                    st.warning("Remember, the performance is counting" \
                    " only the training set,\n so the results are not realistic")


def plot_results():
        """Create an interactive visualization of predictions vs actual values"""
        X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, model = \
        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test, st.session_state.y_train_pred, st.session_state.y_test_pred, st.session_state.model
        if st.session_state.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")
        
        if y_train_pred is None:
            raise ValueError("Model not tested yet. Call test_model() first")
        
        try:
            n_features = X_train.shape[1]
            # Creamos def_fig que siempre se muestra
            def_fig = go.Figure()

            # Convertir a arrays 1D
            y_train_pred = y_train_pred.ravel()
            y_train_actual =  y_train.values.ravel()
                    
            # Puntos train
            def_fig.add_trace(go.Scatter(
                x=y_train_actual, y= y_train_pred,
                mode='markers', name='Train',
                marker=dict(size=5, color='#2E86AB', opacity=0.5)
            ))

            if not st.session_state.trainset_only:
                y_test_pred = y_test_pred.ravel()
                y_test_actual =  y_test.values.ravel()
                def_fig.add_trace(go.Scatter(
                        x=y_test_actual, y= y_test_pred,
                        mode='markers', name='Test',
                        marker=dict(size=6, color='#A23B72',\
                                    opacity=0.6, symbol='x')
                    ))
                all_values = np.concatenate([y_train_actual, y_test_actual,
                                             y_train_pred,  y_test_pred])
            else:
                all_values = np.concatenate([y_train_actual,  y_train_pred])
            
            # Línea perfecta (y = x)
            min_val, max_val = all_values.min(), all_values.max()
            
            # Añadir margen
            margin = (max_val - min_val) * 0.05
            min_val -= margin
            max_val += margin
            
            def_fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect',
                line=dict(color='black', width=2, dash='dash')
            ))
            
            def_fig.update_layout(
                title='Actual vs Predicted Values',
                xaxis_title='Actual', yaxis_title='Predicted',
                template='plotly_white', height=700,
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )

            # Solo creamos fig para casos específicos (1  o 2 features)
            fig = None 

            # CASO 1: Una variable - Gráfico 2D con línea de regresión
            if n_features == 1:
                fig = go.Figure()

                # Puntos de train y test
                fig.add_trace(go.Scatter(
                    x= X_train.iloc[:, 0], y= y_train.iloc[:, 0],
                    mode='markers', name='Train',
                    marker=dict(size=5, color='#2E86AB', opacity=0.6)
                ))

                if not st.session_state.trainset_only:
                    fig.add_trace(go.Scatter(
                                x= X_test.iloc[:, 0], y= y_test.iloc[:, 0],
                                mode='markers', name='Test',
                                marker=dict(size=6, color='#A23B72',\
                                    opacity=0.7, symbol='x')
                            ))
                
                # Línea de regresión
                X_min, X_max =  X_train.iloc[:, 0].min(),  X_train.iloc[:, 0].max()
                X_range = np.linspace(X_min, X_max, 100).reshape(-1, 1)
                y_pred =  model.predict(X_range)
                fig.add_trace(go.Scatter(
                    x=X_range.ravel(), y=y_pred.ravel(),
                    mode='lines', name='Regression',
                    line=dict(color='#F18F01', width=3)
                ))
                
                fig.update_layout(
                    title='Linear Regression: Feature vs Target',
                    xaxis_title= X_train.columns[0],
                    yaxis_title= y_train.columns[0],
                    template='plotly_white', height=600
                )
            
            # CASO 2: Dos variables - Gráfico 3D con plano
            elif n_features == 2:
                fig = go.Figure()

                # Puntos de train
                fig.add_trace(go.Scatter3d(
                    x= X_train.iloc[:, 0], 
                    y= X_train.iloc[:, 1], 
                    z= y_train.iloc[:, 0],
                    mode='markers', name='Train',
                    marker=dict(size=3, color='#2E86AB', opacity=0.7)
                ))
                if not st.session_state.trainset_only:
                    fig.add_trace(go.Scatter3d(
                            x= X_test.iloc[:, 0], 
                            y= X_test.iloc[:, 1], 
                            z= y_test.iloc[:, 0],
                            mode='markers', name='Test',
                            marker=dict(size=4, color='#A23B72',\
                                        opacity=0.8, symbol='x')
                        ))

                # Plano de regresión
                x1_min, x1_max =  X_train.iloc[:, 0].min(),  X_train.iloc[:, 0].max()
                x2_min, x2_max =  X_train.iloc[:, 1].min(),  X_train.iloc[:, 1].max()
                
                x1_range = np.linspace(x1_min, x1_max, 15)
                x2_range = np.linspace(x2_min, x2_max, 15)
                x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
                X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
                z_grid =  model.predict(X_grid).reshape(x1_grid.shape)
                
                fig.add_trace(go.Surface(
                    x=x1_range, y=x2_range, z=z_grid,
                    name='Regression Plane',
                    colorscale='YlOrRd', opacity=0.6,
                    showscale=False
                ))
                
                fig.update_layout(
                    title='3D Linear Regression',
                    scene=dict(
                        xaxis_title= X_train.columns[0],
                        yaxis_title= X_train.columns[1],
                        zaxis_title= y_train.columns[0]
                    ),
                    template='plotly_white', height=700
                )

            # Devolvemos ambas figuras (fig será None si n_features > 2)

            st.plotly_chart(def_fig, use_container_width=True)
            st.divider()
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                st.divider()

        except Exception as e:
            raise RuntimeError(f"Error plotting results: {str(e)}")
        
        
