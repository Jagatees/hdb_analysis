import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset (excluding 2024 data)
df = pd.read_csv("../csv/2017 - 2023.csv")

# Load 2024 data for prediction
df_2024 = pd.read_csv("../csv/2024.csv")

features = [
    'year',
    'month',
    'floor_area_sqm',
    'storey_range_numeric',
    'remaining_lease',
    'score'
]
categorical_features = [
    'town',
    'flat_type',
    'flat_model',
    'region'
]
target = 'resale_price'

# Define preprocessing: scale numerical features and one-hot encode categorical features.
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Create a pipeline for XGBoost with tuned parameters (assumed from GridSearchCV results)
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

# Create a pipeline for RandomForest (base)
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Combine both pipelines into a Voting Regressor
voting_regressor = VotingRegressor(estimators=[
    ('xgb', pipeline_xgb),
    ('rf', pipeline_rf)
])

# Prepare train/test split
X = df[features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Voting Regressor on the training data
voting_regressor.fit(X_train, y_train)

# Evaluate on the test set
y_pred = voting_regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("Voting Regressor Performance:")
print(f"R²: {r2:.4f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Predict prices for the 2024 dataset
X_2024 = df_2024[features + categorical_features]
df_2024['predicted_resale_price'] = voting_regressor.predict(X_2024)

# Calculate overall prediction loss percentage
total_loss = np.sum(np.abs(df_2024['resale_price'] - df_2024['predicted_resale_price']))
total_actual = np.sum(df_2024['resale_price'])
loss_percentage = (total_loss / total_actual) * 100

print(f"Voting Regressor Overall Prediction Loss Percentage: {loss_percentage:.2f}%")

# Save predictions for the 2024 data
df_2024.to_csv("../csv_predicated_model/votiing-regresser.csv", index=False)
print("Predicted resale prices for 2024 saved to votiing-regresser.csv")

# Save model performance metrics to a CSV file
performance_data = {
    "Metric": ["Model", "R²", "MSE", "RMSE", "MAE", "Loss Percentage"],
    "Value": ["VotingRegressor", r2, mse, rmse, mae, loss_percentage]
}
performance_df = pd.DataFrame(performance_data)

# Append performance metrics to existing file if it exists
performance_file = "../model_performance.csv"
try:
    existing_df = pd.read_csv(performance_file)
    performance_df = pd.concat([existing_df, performance_df], ignore_index=True)
except FileNotFoundError:
    pass  # If file does not exist, it will be created

performance_df.to_csv(performance_file, index=False)
print("Model performance metrics appended to model_performance.csv")
