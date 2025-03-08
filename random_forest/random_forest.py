import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import math
import joblib
import time

# Load dataset
df = pd.read_csv("../csv/sampled_hdb_data.csv")

# Define X and Y using the correct columns
X_temp = df[['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'remaining_lease', 'Score', 'region', 'storey_range_numeric']]
y = df['resale_price']

# Handling categorical and numerical columns
cat_cols = ['town', 'flat_type', 'storey_range', 'flat_model', 'region']  # Updated categorical columns
num_cols = ['floor_area_sqm', 'remaining_lease', 'Score', 'storey_range_numeric']  # Updated numerical columns

X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]
X = pd.concat([X_num, X_cat], axis=1)

# Split dataset. 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RobustScaler to handle outliers better
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest with Default Parameters
start_time = time.time()  # Start timing
rf_default = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_default.fit(X_train_scaled, y_train)
elapsed_time = time.time() - start_time  # Time taken to train the model

# Make predictions
y_pred_rf = rf_default.predict(X_test_scaled)

# Evaluate performance
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = math.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Display results
print(f"Random Forest Default Parameters: MSE={mse_rf:.2f}, RMSE={rmse_rf:.2f}, MAE={mae_rf:.2f}, R²={r2_rf:.4f}")
print(f"Model training completed in {elapsed_time:.2f} seconds.")

# Proceed with your GridSearch and other operations as previously
