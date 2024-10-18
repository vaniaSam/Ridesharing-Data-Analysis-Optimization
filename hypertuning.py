import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning with GridSearchCV (NEW CODE)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [10, 20, 30],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples at a leaf node
}

# Create the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Best parameters from the grid search
print("Best parameters found: ", grid_search.best_params_)

# Train the model with the best parameters
best_rf_model = grid_search.best_estimator_

# Step 4: Make predictions on the test set with the best model
y_pred_best_rf = best_rf_model.predict(X_test)

# Step 5: Evaluate the best model
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
rmse_best_rf = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))

print(f"Best Random Forest MAE: {mae_best_rf}")
print(f"Best Random Forest RMSE: {rmse_best_rf}")

