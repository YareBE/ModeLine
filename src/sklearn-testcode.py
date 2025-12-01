from sklearn.linear_model import LinearRegression 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
path = "src/data/housing.csv"  # Execute on main directory (Modeline\)
df = pd.read_csv(path)

# Define features and target
used_features = ['housing_median_age', 'total_rooms', 'total_bedrooms',
                'population', 'median_income']
target = 'median_house_value' 

# Prepare features and target
X = df[used_features]
y = df[target]

# Handle missing values
X = X.fillna(X.mean())

# Normalize features
X = (X - X.mean()) / X.std()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.15, 
                                                   random_state=18)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate errors
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Determination coefficient (R²): {r2:.3f}")  # The closer to 1 the better

print("\nFirst 10 targets vs predictions\n")
print(f"{'Actual_target':<20}{'Predicted_value':<20}\n")
for i in range(10):
    print(f"{y[i]:<20}{y_pred[i]:<20}")

