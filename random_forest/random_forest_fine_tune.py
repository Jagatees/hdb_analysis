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
import os

# Create the output directory for fine-tuned model plots
os.makedirs("random_forest_fine_tune", exist_ok=True)

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

# Hyperparameter tuning via GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 8, 16],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]  # Adding bootstrap to see if it helps with overfitting
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Get the best estimator
best_model = grid_search.best_estimator_

# Make predictions with the tuned model
y_pred_tuned = best_model.predict(X_test_scaled)

# Calculate metrics
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = math.sqrt(mse_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"Random Forest Tuned Parameters: MSE={mse_tuned:.2f}, RMSE={rmse_tuned:.2f}, MAE={mae_tuned:.2f}, R²={r2_tuned:.4f}")

# Save the tuned model
joblib.dump(best_model, 'random_forest_tuned_model.pkl')
print("Tuned model saved as 'random_forest_tuned_model.pkl'")

# ------------------------------
# Plotting for the Tuned Model
# ------------------------------

# 1. Feature Importance Plot
feature_importances_tuned = best_model.feature_importances_
features = X_train.columns
sorted_indices = np.argsort(feature_importances_tuned)[::-1]

plt.figure(figsize=(10, 8))
plt.title('Tuned Model Feature Importances')
plt.bar(range(X_train.shape[1]), feature_importances_tuned[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), features[sorted_indices], rotation=90)
plt.tight_layout()
plt.savefig('random_forest_fine_tune/tuned_feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Actual vs. Predicted Plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_tuned, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Tuned Model: Actual vs. Predicted')
plt.savefig('random_forest_fine_tune/tuned_actual_vs_pred.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Residual Plot
residuals_tuned = y_test - y_pred_tuned
plt.figure(figsize=(10, 8))
plt.scatter(y_pred_tuned, residuals_tuned, alpha=0.5)
plt.hlines(0, y_pred_tuned.min(), y_pred_tuned.max(), colors='red', linestyles='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Tuned Model Residuals Plot')
plt.savefig('random_forest_fine_tune/tuned_residuals.png', dpi=300, bbox_inches='tight')
plt.show()
