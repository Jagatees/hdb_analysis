import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import math

# -----------------------------
# Step A: Load and Prepare Data
# -----------------------------
df = pd.read_csv("../csv/sampled_hdb_data.csv")

X_temp = df[['town', 'flat_type', 'storey_range', 'floor_area_sqm',
             'flat_model', 'remaining_lease', 'Score', 'region', 'storey_range_numeric']]
y = df['resale_price']

cat_cols = ['town', 'flat_type', 'storey_range', 'flat_model', 'region']
num_cols = ['floor_area_sqm', 'remaining_lease', 'Score', 'storey_range_numeric']

# One-hot encode categorical features
X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]
X = pd.concat([X_num, X_cat], axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# Step B: Tune RandomForest and GradientBoost
# --------------------------------------------

# 1. Tune RandomForestRegressor
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid_rf,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid_search_rf.fit(X_train_scaled, y_train)
best_rf = grid_search_rf.best_estimator_
print("Best RandomForest params:", grid_search_rf.best_params_)

# 2. Tune GradientBoostingRegressor
param_grid_gb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [1.0, 0.8]
}

grid_search_gb = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid_gb,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid_search_gb.fit(X_train_scaled, y_train)
best_gb = grid_search_gb.best_estimator_
print("Best GradientBoosting params:", grid_search_gb.best_params_)

# (Optional) Tune LinearRegression or consider a different linear model like Ridge or Lasso
# For demonstration, we'll use the default LinearRegression
best_lr = LinearRegression()

# -----------------------------------
# Step C: Combine into VotingRegressor
# -----------------------------------
voting_regressor = VotingRegressor(estimators=[
    ('rf', best_rf),
    ('gb', best_gb),
    ('lr', best_lr)
])

# Train the VotingRegressor with the best estimators
voting_regressor.fit(X_train_scaled, y_train)

# ----------------------------
# Step D: Evaluate Performance
# ----------------------------
y_pred = voting_regressor.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nVoting Regressor Performance with Tuned Components:")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.4f}")
