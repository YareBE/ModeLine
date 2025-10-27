from sklearn import linear_model

class LRTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self, X_train, y_train):
        """Train the linear regression model"""
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Evaluate the model performance"""
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics = { 
                   'train': {
                        'r2': r2_score(y_train, y_train_pred),
                        'mse': mean_squared_error(y_train, y_train_pred)
                        },
                   'test': {
                        'r2': r2_score(y_test, y_test_pred),
                        'mse': mean_squared_error(y_test, y_test_pred)
                        }
                   }

        return y_train_pred, y_test_pred

    def plot_results(self, y_train, y_train_pred, y_test, y_test_pred):
        """Create a visualizatioin of predictions vs actual values"""
        # Combine data
        train_df = pd.DataFrame({
            'Actual': y_train,
            'Predicted': y_train_pred,
            'Set': 'Train'
        })
        
        test_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_test_pred,
            'Set': 'Test'
        })
        
        combined_df = pd.concat([train_df, test_df])
        
        # Create scatter plot
        fig = px.scatter(
            combined_df,
            x='Actual',
            y='Predicted',
            color='Set',
            title='Predictions vs Actual Values',
            labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
            color_discrete_map={'Train': '#1f77b4', 'Test': '#ff7f0e'}
        )
        
        # Add perfect prediction line
        min_val = min(combined_df['Actual'].min(), combined_df['Predicted'].min())
        max_val = max(combined_df['Actual'].max(), combined_df['Predicted'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        return fig

        
