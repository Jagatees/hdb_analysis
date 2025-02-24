import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("csv/dataset_for_model.csv")
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

# RobustScaler to handle outliers btr 
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning via GridSearchCV to avoid overfitting
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42), 
    param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Cross validate 
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f'Cross-validated R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

# Final evaluation on test set
y_pred = best_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Test new data prediction if needed
"""
new_data = pd.DataFrame({
    'town': ['Bedok'],
    'flat_type': ['3-room'],
    'storey_range': ['04 TO 06'], 
    'floor_area_sqm': [70],
    'flat_model': ['New Generation'],
    'remaining_lease': ['61 years 06 months'],
    'Score': [53],
})

X_newTemp = new_data
cat_temp = X_newTemp.select_dtypes(include=['object']).columns
num_temp = X_newTemp.select_dtypes(exclude=['object']).columns
X_cat_new = pd.get_dummies(X_newTemp[cat_temp], drop_first=True)
X_num_new = X_newTemp[num_temp]
X_new = pd.concat([X_num_new, X_cat_new], axis=1)

# Ensure all columns in X_train are in X_new
X_new_aligned = X_train.columns.to_frame(index=False, name='feature').merge(X_new, how='left', on='feature').fillna(0).drop('feature', axis=1)

new_price_pred = model.predict(scaler.transform(X_new_aligned))

# Output the predicted price
print(f"Predicted price for the new flat: {new_price_pred[0]:.2f}")
"""
