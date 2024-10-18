import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the cleaned data
file_path = 'cleaned_fhv_tripdata_2023-03.csv'
cleaned_data = pd.read_csv(file_path)

# Step 1: Data Preparation

# Extract the hour from the pickup time and add 'day_of_week'
cleaned_data['pickup_datetime'] = pd.to_datetime(cleaned_data['pickup_datetime'])
cleaned_data['pickup_hour'] = cleaned_data['pickup_datetime'].dt.hour
cleaned_data['day_of_week'] = cleaned_data['pickup_datetime'].dt.dayofweek

# Drop rows with missing location IDs or trip duration
model_data = cleaned_data.dropna(subset=['PUlocationID', 'DOlocationID', 'trip_duration_minutes'])

# Define feature matrix (X) and target variable (y)
X = model_data[['PUlocationID', 'DOlocationID', 'pickup_hour', 'day_of_week']]
y = model_data['trip_duration_minutes']

# Step 2: Train the Random Forest Regressor with the best parameters
rf_best_model = RandomForestRegressor(
    max_depth=30,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=300,
    random_state=42
)
rf_best_model.fit(X, y)

# Step 3: Make predictions on the full dataset (for analysis)
y_pred_full = rf_best_model.predict(X)

# Step 4: Evaluate the final model
mae_final = np.mean(np.abs(y - y_pred_full))
rmse_final = np.sqrt(np.mean((y - y_pred_full) ** 2))

print(f"Final Random Forest MAE: {mae_final}")
print(f"Final Random Forest RMSE: {rmse_final}")
