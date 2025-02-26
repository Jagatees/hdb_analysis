import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import math

# Load dataset
df = pd.read_csv("../csv/dataset_for_model.csv")
# ['month', 'town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'remaining_lease', 'resale_price', 'Score']

# Define X and Y
X_temp = df.drop(['resale_price', 'month', 'remaining_lease'], axis=1)
y = df['resale_price']

cat_cols = X_temp.select_dtypes(include=['object']).columns
num_cols = X_temp.select_dtypes(exclude=['object']).columns

X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]
X = pd.concat([X_num, X_cat], axis=1)

# Split dataset. 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RobustScaler to handle outliers better
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a results dictionary to store metrics
results = {}

# Train Random Forest with Default Parameters
print("\n--- Random Forest with Default Parameters ---")
rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_default.predict(X_test_scaled)

# Evaluate performance
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = math.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Results (Default Parameters):")
print(f"MSE: {mse_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"MAE: {mae_rf:.2f}")
print(f"R² Score: {r2_rf:.4f}")

# Store results
results['Random Forest (Default)'] = {
    'MSE': mse_rf,
    'RMSE': rmse_rf,
    'MAE': mae_rf,
    'R²': r2_rf,
    'Parameters': rf_default.get_params()
}

# Cross-validate the default Random Forest model
cv_scores = cross_val_score(rf_default, X_train_scaled, y_train, cv=5, scoring='r2')
print(f'Cross-validated R² (Default RF): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

# Feature importance for Random Forest (default model)
if hasattr(rf_default, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_default.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'].head(15), feature_importance['Importance'].head(15))
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance (Default Model)')
    plt.tight_layout()
    plt.show()

# Hyperparameter tuning via GridSearchCV
print("\n--- Random Forest with Hyperparameter Tuning ---")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

print("Starting hyperparameter tuning. This may take some time...")
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42), 
    param_grid=param_grid, 
    cv=5, 
    scoring='r2', 
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("\nBest Hyperparameters:", grid_search.best_params_)

# Evaluate the tuned model
y_pred_tuned = best_model.predict(X_test_scaled)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = math.sqrt(mse_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"\nRandom Forest Results (Tuned Parameters):")
print(f"MSE: {mse_tuned:.2f}")
print(f"RMSE: {rmse_tuned:.2f}")
print(f"MAE: {mae_tuned:.2f}")
print(f"R² Score: {r2_tuned:.4f}")

# Store results
results['Random Forest (Tuned)'] = {
    'MSE': mse_tuned,
    'RMSE': rmse_tuned,
    'MAE': mae_tuned,
    'R²': r2_tuned,
    'Parameters': grid_search.best_params_
}

# Cross-validate the tuned model
cv_scores_tuned = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f'Cross-validated R² (Tuned RF): {cv_scores_tuned.mean():.3f} ± {cv_scores_tuned.std():.3f}')

# Feature importance for tuned Random Forest model
if hasattr(best_model, 'feature_importances_'):
    feature_importance_tuned = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Tuned Model):")
    print(feature_importance_tuned.head(10))
    
    # Plot feature importance for tuned model
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_tuned['Feature'].head(15), feature_importance_tuned['Importance'].head(15))
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance (Tuned Model)')
    plt.tight_layout()
    plt.show()

# Compare default vs tuned models
print("\n--- Model Comparison ---")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  MSE: {metrics['MSE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  R²: {metrics['R²']:.4f}")
    print()

# Calculate improvement
mse_improvement = ((mse_rf - mse_tuned) / mse_rf) * 100
rmse_improvement = ((rmse_rf - rmse_tuned) / rmse_rf) * 100
mae_improvement = ((mae_rf - mae_tuned) / mae_rf) * 100
r2_improvement = ((r2_tuned - r2_rf) / r2_rf) * 100 if r2_rf > 0 else float('inf')

print("\n--- Improvement After Tuning ---")
print(f"MSE Reduction: {mse_improvement:.2f}%")
print(f"RMSE Reduction: {rmse_improvement:.2f}%")
print(f"MAE Reduction: {mae_improvement:.2f}%")
print(f"R² Improvement: {r2_improvement:.2f}%")

# Save the best model for later use
import joblib
joblib.dump(best_model, 'random_forest_tuned_model.pkl')
print("\nBest model saved as 'random_forest_tuned_model.pkl'")
