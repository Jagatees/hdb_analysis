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
df = pd.read_csv("../csv/flats_with_scores.csv")

# Define X and Y using the correct columns
X_temp = df[['town', 'flat_type', 'block', 'street_name', 'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'remaining_lease', 'Score']]
y = df['resale_price']

# Handling categorical and numerical columns
cat_cols = ['town', 'flat_type', 'block', 'street_name', 'storey_range', 'flat_model', 'lease_commence_date', 'remaining_lease']  # Assuming these are categorical
num_cols = ['floor_area_sqm', 'Score']  # Assuming these are numerical

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

# Hyperparameter tuning via GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42, n_jobs=-1), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best model and evaluation
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test_scaled)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = math.sqrt(mse_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"Random Forest Tuned Parameters: MSE={mse_tuned:.2f}, RMSE={rmse_tuned:.2f}, MAE={mae_tuned:.2f}, R²={r2_tuned:.4f}")

# Save the tuned model
joblib.dump(best_model, 'random_forest_tuned_model.pkl')
print("Tuned model saved as 'random_forest_tuned_model.pkl'")

# Visualization of results
metrics = ['MSE', 'RMSE', 'MAE', 'R²']
default_values = [mse_rf, rmse_rf, mae_rf, r2_rf]
tuned_values = [mse_tuned, rmse_tuned, mae_tuned, r2_tuned]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, default_values, width, label='Default')
rects2 = ax.bar(x + width/2, tuned_values, width, label='Tuned')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by model and metric')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

fig.tight_layout()
plt.show()
