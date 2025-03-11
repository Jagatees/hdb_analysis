import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load datasets
df = pd.read_csv("../csv/sampled_hdb_no2024_data.csv")
df_2024 = pd.read_csv("../csv/sampled_hdb_2024_data.csv")

# Define features and target variable
features = ['year', 'month', 'flat_age', 'floor_area_sqm', 
            'storey_range_numeric', 'price_per_square_meter', 'remaining_lease']
categorical_features = ['town', 'flat_type', 'flat_model', 'region']
target = 'resale_price'

# Preprocessing: standardize numerical features and one-hot encode categorical features.
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Create individual pipelines for base models

# Pipeline for tuned XGBoost (example tuned parameters)
pipeline_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        n_estimators=200,       # example tuned value
        max_depth=5,            # example tuned value
        learning_rate=0.1,      # example tuned value
        subsample=0.8,          # example tuned value
        colsample_bytree=0.8    # example tuned value
    ))
])

# Pipeline for RandomForest (base)
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Define base estimators for stacking
estimators = [
    ('xgb', pipeline_xgb),
    ('rf', pipeline_rf)
]

# Define the Stacking Regressor with a Ridge meta-model
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

# Prepare training and testing sets
X = df[features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the stacking regressor
stacking_regressor.fit(X_train, y_train)

# Evaluate on the test set
y_pred = stacking_regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("Stacking Regressor Performance:")
print(f"R²: {r2:.4f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Predict prices for the 2024 dataset
X_2024 = df_2024[features + categorical_features]
df_2024['predicted_resale_price'] = stacking_regressor.predict(X_2024)

# Calculate overall loss percentage
total_loss = np.sum(np.abs(df_2024['resale_price'] - df_2024['predicted_resale_price']))
total_actual = np.sum(df_2024['resale_price'])
loss_percentage = (total_loss / total_actual) * 100

print(f"Stacking Regressor Overall Prediction Loss Percentage: {loss_percentage:.2f}%")

# Save predictions for the 2024 data
df_2024.to_csv("./sampled_hdb_2024_predictions_StackingRegressor.csv", index=False)
print("Predicted resale prices for 2024 saved to sampled_hdb_2024_predictions_StackingRegressor.csv")
