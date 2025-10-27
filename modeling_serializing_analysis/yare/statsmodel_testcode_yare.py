import pandas as pd
import numpy as np
import statsmodels.api as sm
from time import perf_counter_ns

t1 = perf_counter_ns()
# Load and prepare data
path = "src/data/housing.csv"
df = pd.read_csv(path)

used_features = ['housing_median_age', 'total_rooms', 'total_bedrooms',
                 'population', 'median_income']
target = 'median_house_value'

# Prepare features and target
X = df[used_features].copy()
y = df[target].copy()

# Handle missing values and normalize features
X = X.fillna(X.mean())
X = (X - X.mean()) / X.std()

# Split data into train (85%) and test (15%)
np.random.seed(18)
msk = np.random.rand(len(df)) < 0.85
X_train = X[msk]
X_test = X[~msk]
y_train = y[msk]
y_test = y[~msk]

# Add constant term and fit model
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
model = sm.OLS(y_train, X_train)
results = model.fit()

# Calculate predictions and metrics
y_pred = results.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Show first 10 predictions vs actual values
X_sample = sm.add_constant(X.iloc[:10])
predictions = results.predict(X_sample)
print("\nFirst 10 houses:")
comparison = pd.DataFrame({
    'Predicted': predictions,
    'Actual': y.iloc[:10]
}).round(2)
print(comparison)

t2 = perf_counter_ns()

print("Time spent with statsmodels in ms: ", (t2-t1)/1000000)