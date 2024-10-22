import pandas as pd

# Load the CSV file with a different encoding
file_path = 'fhv_tripdata_2023-03.csv'
fhv_data_csv = pd.read_csv(file_path, encoding='ISO-8859-1')

#  Data Cleaning

# Convert pickup and dropoff datetime columns to datetime format, handling errors
fhv_data_csv['pickup_datetime'] = pd.to_datetime(fhv_data_csv['pickup_datetime'], errors='coerce')
fhv_data_csv['dropOff_datetime'] = pd.to_datetime(fhv_data_csv['dropOff_datetime'], errors='coerce')

# Drop rows where both pickup and dropoff location IDs are missing, and create a copy of the data
cleaned_data = fhv_data_csv.dropna(subset=['PUlocationID', 'DOlocationID'], how='all').copy()

# Feature Engineering

# Calculate trip duration (in minutes) by subtracting pickup from dropoff datetime
cleaned_data['trip_duration_minutes'] = (cleaned_data['dropOff_datetime'] - cleaned_data['pickup_datetime']).dt.total_seconds() / 60

#  Data Analysis

# Summary statistics for trip duration
trip_duration_stats = cleaned_data['trip_duration_minutes'].describe()
print("Trip Duration Stats:")
print(trip_duration_stats)

# Analyze top 5 most frequent pickup and dropoff locations
top_pickup_locations = cleaned_data['PUlocationID'].value_counts().head(5)
top_dropoff_locations = cleaned_data['DOlocationID'].value_counts().head(5)

print("\nTop 5 Pickup Locations:")
print(top_pickup_locations)

print("\nTop 5 Dropoff Locations:")
print(top_dropoff_locations)

cleaned_data.to_csv('cleaned_fhv_tripdata_2023-03.csv', index=False)
