import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
# df = pd.read_csv("dataset_for_model.csv")
df = pd.read_csv("sampled_hdb_data.csv")

# Define X and Y
# X_temp = df.drop(['resale_price', 'month', 'town', 'storey_range'], axis=1)
X_temp = df[['year', 'flat_type', 'floor_area_sqm', 'remaining_lease', 'Score', 'storey_range_numeric', 'region']]
y = df['resale_price']

# Extract numerical and categorical values
cat_cols = X_temp.select_dtypes(include=['object']).columns
num_cols = X_temp.select_dtypes(exclude=['object']).columns
X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]
X = pd.concat([X_num, X_cat], axis=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols)
    ],
    remainder='passthrough'  # This leaves the one-hot columns unchanged
)
X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

# Hyperparameter grid
param_grid = {
    'n_neighbors': list(range(1, 21)),  # K-values from 1 to 20
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'weights': ['uniform', 'distance']
}

# Train KNN model
model = KNeighborsRegressor()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Show best params
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Train best model
best_knn = grid_search.best_estimator_

# Predict target values for the normalized test set
y_pred = best_knn.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Default Model:")
print(f"MSE: {mse:.2f}, RMSE {rmse:.2f}, MAE {mae:.2f}, R²: {r2:.2f}",)
