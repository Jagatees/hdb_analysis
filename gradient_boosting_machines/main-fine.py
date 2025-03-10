# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import RobustScaler
# import xgboost as xgb
# import math

# # Load dataset
# df = pd.read_csv("../csv/sampled_hdb_data.csv")

# # Define X and Y
# X_temp = df[['town', 'flat_type', 'storey_range', 'floor_area_sqm',
#              'flat_model', 'remaining_lease', 'Score', 'region', 'storey_range_numeric']]
# y = df['resale_price']

# # One-hot encoding categorical columns
# cat_cols = ['town', 'flat_type', 'storey_range', 'flat_model', 'region']
# num_cols = ['floor_area_sqm', 'remaining_lease', 'Score', 'storey_range_numeric']
# X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
# X_num = X_temp[num_cols]
# X = pd.concat([X_num, X_cat], axis=1)

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale features using RobustScaler
# scaler = RobustScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Initialize the XGBRegressor
# xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [3, 6, 9],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'subsample': [0.8, 1],
#     'colsample_bytree': [0.8, 1]
# }

# # Create the GridSearchCV object without early stopping parameters
# grid_search = GridSearchCV(
#     estimator=xgb_reg,
#     param_grid=param_grid,
#     cv=3,
#     scoring='neg_mean_squared_error',
#     verbose=1,
#     n_jobs=-1
# )

# # Fit GridSearchCV without early stopping
# grid_search.fit(X_train_scaled, y_train)

# # Output best parameters and best score
# print("Best parameters found:", grid_search.best_params_)
# print("Best CV score (neg MSE):", grid_search.best_score_)

# # Evaluate the best model on the test set
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test_scaled)
# mse = mean_squared_error(y_test, y_pred)
# rmse = math.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Best XGBoost Regressor MSE: {mse:.2f}")
# print(f"Best XGBoost Regressor RMSE: {rmse:.2f}")
# print(f"Best XGBoost Regressor MAE: {mae:.2f}")
# print(f"Best XGBoost Regressor R²: {r2:.4f}")


# this shit crash my computer XD