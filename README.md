Project Overview

This project analyzes and optimizes ridesharing data from New York City's For-Hire Vehicle (FHV) trip data for March 2023. The main objectives of the project include:

Data Cleaning: 
Processing raw trip data to remove inconsistencies and prepare the data for analysis.

Feature Engineering: 
Creating new features, such as trip duration and time-related variables.

Modeling: 
Predicting trip duration using machine learning models, including Random Forest Regressor, with hyperparameter tuning for optimal performance.

Visualization: 
Visualizing the results of the analysis and the model's performance.

The project workflow involved data preprocessing, building and evaluating machine learning models, hyperparameter tuning, and visualizing the results.

Project Structure

main.py: Contains the data cleaning and feature engineering steps.
main2.py: Contains the machine learning model development using a Random Forest Regressor, including hyperparameter tuning for optimized performance.
visualization.py: Visualizes the performance of the models, including MAE and RMSE metrics.
Data Source

The dataset used for this project is the NYC FHV Trip Data for March 2023, downloaded from the NYC Taxi & Limousine Commission. This dataset contains detailed information about for-hire vehicle trips in New York City, including pickup and dropoff locations and timestamps.

Raw Data: fhv_tripdata_2023-03.csv (77.86 MB)
Cleaned Data: cleaned_fhv_tripdata_2023-03.csv (74.26 MB)
Note: The raw and cleaned data files were not included in this repository due to GitHub’s file size limits. They can be regenerated using the steps outlined in this project.
Key Steps

1. Data Cleaning and Feature Engineering (main.py)
Converted pickup and dropoff timestamps to datetime format.
Removed rows with missing pickup and dropoff locations.
Calculated trip duration in minutes from pickup and dropoff times.
Extracted new features: pickup_hour and day_of_week.
2. Model Development, Training, and Hyperparameter Tuning (main2.py)
Split the dataset into training (80%) and testing (20%) sets.
Trained a Random Forest Regressor model to predict trip duration based on features such as pickup location, dropoff location, time of day, and day of the week.
Evaluated the model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
Performed hyperparameter tuning directly within main2.py using GridSearchCV to optimize the Random Forest model.
Model Performance:

Initial Random Forest MAE: 120.497
Initial Random Forest RMSE: 1526.91
Tuned Model Performance:

Final Random Forest MAE: 77.176
Final Random Forest RMSE: 1015.509
3. Visualization of Results (visualization.py)
Generated visualizations comparing predicted vs. actual trip durations.
Created performance plots for MAE and RMSE to assess the effectiveness of the model.
Tools and Technologies

Python: The core programming language used for data processing and model development.
Pandas: Used for data manipulation and cleaning.
NumPy: Used for numerical operations and performance metrics.
Scikit-learn: Utilized for machine learning model development, evaluation, and hyperparameter tuning.
Matplotlib & Seaborn: Libraries used to create visualizations.
GridSearchCV: Used to tune hyperparameters of the Random Forest model.

Prerequisites:
Ensure you have Python 3.x installed and the required libraries. Install dependencies via pip

HOW TO RUN

Data Cleaning and Feature Engineering: Run the main.py script to clean the data and create new features.

Model Training and Hyperparameter Tuning: Run main2.py to train the Random Forest model on the cleaned data, including hyperparameter tuning

Visualization: Run the visualization script to generate plots showing the model’s predictions vs actual data

Results and Conclusion

The project successfully developed and optimized a Random Forest Regressor to predict trip durations with a final MAE of 77.176 minutes and an RMSE of 1015.509 minutes. The visualizations helped in understanding the relationship between actual and predicted trip durations, revealing opportunities for further feature engineering or model tuning.

Future Work

Integrating additional features like weather or traffic conditions to improve the model’s predictive power.
Implementing more advanced models, such as Gradient Boosting or XGBoost.
Automating data collection and model retraining to adapt to changes in the data over time.


