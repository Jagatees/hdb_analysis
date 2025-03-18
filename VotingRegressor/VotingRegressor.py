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
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("Voting Regressor Performance:")
print(f"R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

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
print("Predicted resale prices for 2024 saved to sampled_hdb_2024_predictions_VotingRegressor.csv")



# Function to evaluate model and save results
def evaluate_and_save_model(model_name, y_true, y_pred, filename="../model_performance.csv"):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate Prediction Loss Percentage
    total_loss = np.sum(np.abs(y_true - y_pred))
    total_actual = np.sum(y_true)
    loss_percentage = (total_loss / total_actual) * 100

    # Store results in DataFrame
    results = pd.DataFrame({
        "Model": [model_name],
        "R² Score": [r2],
        "RMSE": [rmse],
        "MSE": [mse],
        "MAE": [mae],
        "Prediction Loss %": [loss_percentage]
    })

    # Append or create file
    try:
        existing_results = pd.read_csv(filename)
        results = pd.concat([existing_results, results], ignore_index=True)
    except FileNotFoundError:
        pass  # First time writing file

    results.to_csv(filename, index=False)
    print(f"Results for {model_name} saved to {filename}")

    evaluate_and_save_model("Voting Regressor", y_test, y_pred)


