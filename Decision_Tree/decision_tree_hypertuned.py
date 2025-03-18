import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset (excluding 2024 data)
df = pd.read_csv("../csv/2017 - 2023.csv")

# Load 2024 data for prediction
df_2024 = pd.read_csv("../csv/2024.csv")

# Define features and target variable
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

# One-Hot Encoding for categorical features and scaling for numerical features
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Define the Decision Tree pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Define hyperparameter grid for tuning the decision tree
param_grid = {
    'regressor__max_depth': [None, 5, 10, 15, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

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

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Decision Tree - R²: {r2:.4f}, RMSE: {rmse:.2f}, MSE: {mse:.2f}, MAE: {mae:.2f}")

# Predict prices for the 2024 dataset
X_2024 = df_2024[features + categorical_features]
df_2024['predicted_resale_price'] = best_model.predict(X_2024)

# Calculate the overall loss percentage
total_loss = np.sum(np.abs(df_2024['resale_price'] - df_2024['predicted_resale_price']))
total_actual = np.sum(df_2024['resale_price'])
loss_percentage = (total_loss / total_actual) * 100

print(f"Overall Prediction Loss Percentage: {loss_percentage:.2f}%")

# Save predictions for the 2024 data
df_2024.to_csv("../csv_predicated_model/DecisionTree_GridSearch.csv", index=False)
print("Predicted resale prices for 2024 saved to DecisionTree_GridSearch.csv")


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

# Example usage in your model script
evaluate_and_save_model("Decision Tree GridSearchCV", y_test, y_pred)