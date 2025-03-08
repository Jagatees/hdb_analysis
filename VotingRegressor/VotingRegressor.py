import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

# Load dataset
df = pd.read_csv("../csv/sampled_hdb_data.csv")

# Define X and Y
X_temp = df[['town', 'flat_type', 'storey_range', 'floor_area_sqm',
             'flat_model', 'remaining_lease', 'Score', 'region', 'storey_range_numeric']]
y = df['resale_price']

# One-hot encoding categorical columns
cat_cols = ['town', 'flat_type', 'storey_range', 'flat_model', 'region']
num_cols = ['floor_area_sqm', 'remaining_lease', 'Score', 'storey_range_numeric']
X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]
X = pd.concat([X_num, X_cat], axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize individual regressors
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
lr = LinearRegression()

# Create the Voting Regressor
voting_regressor = VotingRegressor(estimators=[
    ('rf', rf),
    ('gb', gb),
    ('lr', lr)
])

# Train the Voting Regressor
voting_regressor.fit(X_train_scaled, y_train)

# Make predictions
y_pred = voting_regressor.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print(f"Voting Regressor MSE: {mse:.2f}")
print(f"Voting Regressor RMSE: {rmse:.2f}")
print(f"Voting Regressor MAE: {mae:.2f}")
print(f"Voting Regressor R²: {r2:.4f}")
