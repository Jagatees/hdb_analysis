import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("dataset_for_model.csv")

# Define X and Y
X_temp = df.drop(['resale_price', 'month', 'remaining_lease'], axis=1)
y = df['resale_price']

# Identify categorical and numerical columns
cat_cols = X_temp.select_dtypes(include=['object']).columns
num_cols = X_temp.select_dtypes(exclude=['object']).columns

# One-Hot Encoding for categorical columns
X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]

# Merge categorical and numerical features
X = pd.concat([X_num, X_cat], axis=1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Hyperparameter Grid for GridSearchCV
param_grid = {
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [2, 5, 15, 30],
    'min_samples_leaf': [1, 5, 10],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1, 1.0]
}

# Initialize Decision Tree Model
dt_model = DecisionTreeRegressor(random_state=42)

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    dt_model,
    param_grid,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Train Model with GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Get Best Model
best_model = grid_search.best_estimator_

# Make Predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluate Model Performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"MSE: {mse:.2f}, RMSE, {rmse:.2f}, MAE,{mae:.2f}, R²: {r2:.2f}",)
print("Best Parameters:", grid_search.best_params_)

# Get Feature Importance from the best model
feature_importance = best_model.feature_importances_

# Create a DataFrame to show the feature importance alongside the feature names
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})

# Sort the DataFrame by importance (in descending order)
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the top 10 most important features
top_10_importance = importance_df.head(10)
print(top_10_importance)



# Save Best Model and Scaler for Streamlit Integration
joblib.dump(best_model, "decision_tree_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Decision Tree model and scaler saved successfully!")

'''
# Test with New Flat Data
new_data = pd.DataFrame({
    'month': ['2017-01'],
    'town': ['Bedok'],
    'flat_type': ['3-room'],
    'storey_range': ['04 TO 06'],
    'floor_area_sqm': [70],
    'flat_model': ['New Generation'],
    'remaining_lease': ['61 years 06 months'],
    'resale_price': [280000],  # Not used for prediction
    'Score': [53],
})

# Prepare New Data for Prediction
X_newTemp = new_data.drop(['resale_price', 'month'], axis=1)
cat_temp = X_newTemp.select_dtypes(include=['object']).columns
num_temp = X_newTemp.select_dtypes(exclude=['object']).columns

# One-Hot Encode the new data
X_cat_new = pd.get_dummies(X_newTemp[cat_temp], drop_first=True)
X_num_new = X_newTemp[num_temp]
X_new = pd.concat([X_num_new, X_cat_new], axis=1)

# Ensure the input matches training data columns
missing_cols = set(X.columns) - set(X_new.columns)
for col in missing_cols:
    X_new[col] = 0  # Add missing categorical columns

X_new = X_new[X.columns]  # Reorder columns to match training

# Scale New Data
X_new_scaled = scaler.transform(X_new)

# Predict New Flat Price
new_price_pred = best_model.predict(X_new_scaled)
print(f"Predicted price for the new flat: {new_price_pred[0]:,.2f}")
'''