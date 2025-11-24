from sklearn.linear_model import LinearRegression 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from sklearn.pipeline import Pipeline
from time import perf_counter_ns


MODEL_PATH = "model_packet.joblib"

# Load data
DATA_PATH = "src/data/housing.csv"  # Execute on main directory (Modeline\)

t1 = perf_counter_ns()
if os.path.exists(MODEL_PATH):
    # DESERIALIZE
    loaded_pipe, X_loaded, y_test = joblib.load(MODEL_PATH)
    print("\nWITH DESERIALIZED MODEL\n")
    # Make predictions
    y_pred = loaded_pipe.predict(X_loaded)

else:
    df = pd.read_csv(DATA_PATH)

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
    pipe = Pipeline([
    ('reg', LinearRegression())
    ])
    pipe.fit(X_train, y_train)

    # Make predictions
    y_pred = pipe.predict(X_test)

    #SERIALIZE WITH JOBLIB AND STORE X_TEST/Y_TEST
    joblib.dump((pipe, X_test, y_test), MODEL_PATH)
    
    print("\nBEFORE SERIALIZING\n")

print(f"{'Actual_target':<20}{'Predicted_value':<20}\n")
for i in range(10):
    print(f"{y_test.iloc[i]:<20}{y_pred[i]:<20}")

# Calculate errors
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Determination coefficient (R²): {r2:.3f}")

t2 = perf_counter_ns()

print("Time spent in ms: ", (t2-t1)//1000000)

