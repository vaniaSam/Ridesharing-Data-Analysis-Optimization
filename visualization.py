import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Make predictions on the full dataset
y_pred_full = rf_best_model.predict(X)

# Visualization 1: Actual vs Predicted Trip Duration
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred_full, alpha=0.3)
plt.title('Actual vs Predicted Trip Duration')
plt.xlabel('Actual Trip Duration (minutes)')
plt.ylabel('Predicted Trip Duration (minutes)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line for reference
plt.show()

# Visualization 2: Error Distribution (Residuals)
residuals = y - y_pred_full
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.title('Residuals (Actual - Predicted) Distribution')
plt.xlabel('Error (minutes)')
plt.ylabel('Frequency')
plt.show()

# Visualization 3: Feature Importance
feature_importances = rf_best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.show()
