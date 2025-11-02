from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import plotly.graph_objects as go


class LRTrainer:
    
    def __init__(self, X, y, train_size):
        """
        Initialize the Linear Regression Trainer
        
        Args:
            X: Features (DataFrame or array-like)
            y: Target variable (DataFrame or array-like)
            train_size: Size of training set (float for proportion, int for absolute number)
        """
        self.model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self._test_available = True if len(X) >= 10 else False
        
        # Validaciones
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got X: {len(X)}, y: {len(y)}")
        
        if len(X) == 0:
            raise ValueError("X and y cannot be empty")
        
        assert (0 <= train_size <= 1), "The training size must be between 0 and 1"
        
        # Convertir a DataFrame si es necesario
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        self._split_dataset(X, y, train_size)
    
    def _split_dataset(self, X, y, train_size: int):
        """Split dataset into train and test sets"""
        if self._test_available:
            self.X_train, self.X_test, self.y_train, \
            self.y_test = train_test_split(X, y, train_size=train_size,
                                            random_state=18)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test =\
                X, None, y, None
        
        
    def train_model(self):
        """Train the linear regression model"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not initialized. Check __init__ method")
        
        try:
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            return self.model
        except Exception as e:
            raise RuntimeError(f"Error training model: {str(e)}")
    
    def get_splitted_subsets(self):
        """Get the train/test subsets"""
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_formula(self):
        """Get the regression formula with feature names"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")
        
        try:
            # Obtener nombres de variables
            feature_names = self.X_train.columns.tolist()
            target_name = self.y_train.columns[0]
            
            # Construir fórmula
            coefs = self.model.coef_.ravel()
            intercept = self.model.intercept_[0]
            
            formula_parts = [f"{target_name} = "]
            
            for i, (name, coef) in enumerate(zip(feature_names, coefs)):
                if i == 0:
                    formula_parts.append(f"{coef:.4f} * {name}")
                else:
                    sign = "+" if coef >= 0 else "-"
                    formula_parts.append(f" {sign} {abs(coef):.4f} * {name}")
            
            # Añadir intercept
            sign = "+" if intercept >= 0 else "-"
            formula_parts.append(f" {sign} {abs(intercept):.4f}")
            
            return "".join(formula_parts)
        except Exception as e:
            raise RuntimeError(f"Error generating formula: {str(e)}")
    
    def test_model(self):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")

        try:
            y_test_pred = None
            y_train_pred = self.model.predict(self.X_train)
            metrics = {
                'train': {
                    'r2': r2_score(self.y_train, y_train_pred),
                    'mse': mean_squared_error(self.y_train, y_train_pred),
                    'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                    'mae': mean_absolute_error(self.y_train, y_train_pred)
                }     
            }
            self.y_train_pred = y_train_pred.ravel()

            if self._test_available:
                y_test_pred = self.model.predict(self.X_test)
                metrics['test'] = {
                        'r2': r2_score(self.y_test, y_test_pred),
                        'mse': mean_squared_error(self.y_test, y_test_pred),
                        'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                        'mae': mean_absolute_error(self.y_test, y_test_pred)
                    }
                self.y_test_pred = y_test_pred.ravel()

            return metrics, y_train_pred, y_test_pred
  
            
        except Exception as e:
            raise RuntimeError(f"Error testing model: {str(e)}")

    def plot_results(self):
        """Create an interactive visualization of predictions vs actual values"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first")
        
        if self.y_train_pred is None:
            raise ValueError("Model not tested yet. Call test_model() first")
        
        try:
            n_features = self.X_train.shape[1]
            fig = go.Figure()
            
            # CASO 1: Una variable - Gráfico 2D con línea de regresión
            if n_features == 1:
                # Puntos de train y test
                fig.add_trace(go.Scatter(
                    x=self.X_train.iloc[:, 0], y=self.y_train.iloc[:, 0],
                    mode='markers', name='Train',
                    marker=dict(size=5, color='#2E86AB', opacity=0.6)
                ))

                if self._test_available:
                    fig.add_trace(go.Scatter(
                                x=self.X_test.iloc[:, 0], y=self.y_test.iloc[:, 0],
                                mode='markers', name='Test',
                                marker=dict(size=6, color='#A23B72',\
                                    opacity=0.7, symbol='x')
                            ))
                
                # Línea de regresión
                X_min, X_max = self.X_train.iloc[:, 0].min(), self.X_train.iloc[:, 0].max()
                X_range = np.linspace(X_min, X_max, 100).reshape(-1, 1)
                y_pred = self.model.predict(X_range)
                fig.add_trace(go.Scatter(
                    x=X_range.ravel(), y=y_pred.ravel(),
                    mode='lines', name='Regression',
                    line=dict(color='#F18F01', width=3)
                ))
                
                fig.update_layout(
                    title='Linear Regression: Feature vs Target',
                    xaxis_title=self.X_train.columns[0],
                    yaxis_title=self.y_train.columns[0],
                    template='plotly_white', height=600
                )
            
            # CASO 2: Dos variables - Gráfico 3D con plano
            elif n_features == 2:
                # Puntos de train
                fig.add_trace(go.Scatter3d(
                    x=self.X_train.iloc[:, 0], 
                    y=self.X_train.iloc[:, 1], 
                    z=self.y_train.iloc[:, 0],
                    mode='markers', name='Train',
                    marker=dict(size=3, color='#2E86AB', opacity=0.7)
                ))
                if self._test_available:
                    fig.add_trace(go.Scatter3d(
                            x=self.X_test.iloc[:, 0], 
                            y=self.X_test.iloc[:, 1], 
                            z=self.y_test.iloc[:, 0],
                            mode='markers', name='Test',
                            marker=dict(size=4, color='#A23B72',\
                                        opacity=0.8, symbol='x')
                        ))

                # Plano de regresión
                x1_min, x1_max = self.X_train.iloc[:, 0].min(), self.X_train.iloc[:, 0].max()
                x2_min, x2_max = self.X_train.iloc[:, 1].min(), self.X_train.iloc[:, 1].max()
                
                x1_range = np.linspace(x1_min, x1_max, 15)
                x2_range = np.linspace(x2_min, x2_max, 15)
                x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
                X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
                z_grid = self.model.predict(X_grid).reshape(x1_grid.shape)
                
                fig.add_trace(go.Surface(
                    x=x1_range, y=x2_range, z=z_grid,
                    name='Regression Plane',
                    colorscale='YlOrRd', opacity=0.6,
                    showscale=False
                ))
                
                fig.update_layout(
                    title='3D Linear Regression',
                    scene=dict(
                        xaxis_title=self.X_train.columns[0],
                        yaxis_title=self.X_train.columns[1],
                        zaxis_title=self.y_train.columns[0]
                    ),
                    template='plotly_white', height=700
                )
            
            # CASO 3: Más variables - Actual vs Predicted
            else:
                # Convertir a arrays 1D
                y_train_actual = self.y_train.values.ravel()
                    
                # Puntos train
                fig.add_trace(go.Scatter(
                    x=y_train_actual, y=self.y_train_pred,
                    mode='markers', name='Train',
                    marker=dict(size=5, color='#2E86AB', opacity=0.5)
                ))

                if self._test_available:
                    y_test_actual = self.y_test.values.ravel()
                    fig.add_trace(go.Scatter(
                            x=y_test_actual, y=self.y_test_pred,
                            mode='markers', name='Test',
                            marker=dict(size=6, color='#A23B72',\
                                        opacity=0.6, symbol='x')
                        ))
                    all_values = np.concatenate([y_train_actual, y_test_actual,
                                                self.y_train_pred, self.y_test_pred])
                else:
                    all_values = np.concatenate([y_train_actual, self.y_train_pred])
                
                # Línea perfecta (y = x)
                min_val, max_val = all_values.min(), all_values.max()
                
                # Añadir margen
                margin = (max_val - min_val) * 0.05
                min_val -= margin
                max_val += margin
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', name='Perfect',
                    line=dict(color='black', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Actual vs Predicted Values',
                    xaxis_title='Actual', yaxis_title='Predicted',
                    template='plotly_white', height=700,
                    yaxis=dict(scaleanchor="x", scaleratio=1)
                )

            return fig
            
        except Exception as e:
            raise RuntimeError(f"Error plotting results: {str(e)}")