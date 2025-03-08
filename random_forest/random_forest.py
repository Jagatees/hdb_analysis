import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import time
import os

# Optional: Create the "images" folder if it doesn't exist
os.makedirs("random_forest", exist_ok=True)

# Load dataset and preprocess
df = pd.read_csv("../csv/sampled_hdb_data.csv")
X_temp = df[['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'remaining_lease', 'Score', 'region', 'storey_range_numeric']]
y = df['resale_price']

cat_cols = ['town', 'flat_type', 'storey_range', 'flat_model', 'region']
num_cols = ['floor_area_sqm', 'remaining_lease', 'Score', 'storey_range_numeric']

X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]
X = pd.concat([X_num, X_cat], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate model
rf_default = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_default.fit(X_train_scaled, y_train)
y_pred_rf = rf_default.predict(X_test_scaled)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = math.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Default Parameters: MSE={mse_rf:.2f}, RMSE={rmse_rf:.2f}, MAE={mae_rf:.2f}, R²={r2_rf:.4f}")

# 1. Feature Importance Plot
feature_importances = rf_default.feature_importances_
features = X_train.columns
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), features[sorted_indices], rotation=90)
plt.tight_layout()
plt.savefig('random_forest/random_forest_base_feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Actual vs. Predicted Plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.savefig('random_forest/random_forest_base_actual_vs_pred.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Residual Plot
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 8))
plt.scatter(y_pred_rf, residuals, alpha=0.5)
plt.hlines(0, y_pred_rf.min(), y_pred_rf.max(), colors='red', linestyles='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.savefig('random_forest/random_forest_base_residuals.png', dpi=300, bbox_inches='tight')
plt.show()
