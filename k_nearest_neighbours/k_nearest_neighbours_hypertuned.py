import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset (excluding 2024 data)
df = pd.read_csv("../csv/2017 - 2023.csv")

# Load 2024 data for prediction
df_2024 = pd.read_csv("../csv/2024.csv")

# Define features and target variable
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

# Preprocessing: Standardize numerical features and one-hot encode categorical features
# Use sparse_output=False to get a dense array.
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Define the pipeline with KNeighborsRegressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor())
])

# Define hyperparameter grid for KNeighborsRegressor
param_grid = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11],
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2]  # 1 for Manhattan, 2 for Euclidean distance
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

# Prepare training and testing sets
X = df[features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Display best parameters and corresponding score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score (negative MSE):", grid_search.best_score_)

# Use the best estimator to evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"KNN with GridSearch - R²: {r2:.4f}, RMSE: {rmse:.2f}, MSE: {mse:.2f}, MAE: {mae:.2f}")

# Predict prices for the 2024 dataset
X_2024 = df_2024[features + categorical_features]
df_2024['predicted_resale_price'] = best_model.predict(X_2024)

# Calculate overall loss percentage
total_loss = np.sum(np.abs(df_2024['resale_price'] - df_2024['predicted_resale_price']))
total_actual = np.sum(df_2024['resale_price'])
loss_percentage = (total_loss / total_actual) * 100

print(f"Overall Prediction Loss Percentage: {loss_percentage:.2f}%")

# Save predictions for the 2024 data
df_2024.to_csv("../csv_predicated_model/KNN_GridSearch.csv", index=False)
print("Predicted resale prices for 2024 saved KNN_GridSearch.csv")


# Save model performance metrics to a CSV file
performance_data = {
    "Metric": ["Model","R²", "RMSE", "MSE", "MAE", "Loss Percentage"],
    "Value": ["KNN with GridSearch",r2, rmse, mse, mae, loss_percentage]
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