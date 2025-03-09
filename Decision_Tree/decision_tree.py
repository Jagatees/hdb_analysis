import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("../csv/sampled_hdb_data.csv")
# ['month','town','flat_type','storey_range','floor_area_sqm','flat_model', 'remaining_lease', 'resale_price','Score']

# Define X and Y
X_temp = df[['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'remaining_lease', 'Score', 'region', 'storey_range_numeric']]
y = df['resale_price']
cat_cols = X_temp.select_dtypes(include=['object']).columns
num_cols = X_temp.select_dtypes(exclude=['object']).columns
X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]
X = pd.concat([X_num, X_cat], axis=1)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}",)


# Get feature importance scores
feature_importance = model.feature_importances_

# Feature Importance
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.to_csv("feature_importance.csv", index=False)
plt.figure(figsize=(10, 15))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'], palette='viridis')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.savefig("base_graph/decision_tree_feature_importance.png", dpi=300, bbox_inches="tight")

# Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.axline((0, 0), slope=1, color="red", linestyle="dashed")  # Diagonal reference line
plt.savefig("base_graph/decision_tree_actual_vs_predicted.png", dpi=300, bbox_inches="tight")

# Test
"""
new_data = pd.DataFrame({
    'month': ['2017-01'],
    'town': ['Bedok'],
    'flat_type': ['3-room'],
    'storey_range': ['04 TO 06'],
    'floor_area_sqm': [70],
    'flat_model': ['New Generation'],
    'remaining_lease': ['61 years 06 months'],
    'resale_price': [280000],
    'Score': [53],
})

X_newTemp = new_data.drop(['resale_price', 'month'], axis=1)
y_new = df['resale_price']
cat_temp = X_newTemp.select_dtypes(include=['object']).columns
num_temp = X_newTemp.select_dtypes(exclude=['object']).columns
X_cat = pd.get_dummies(X_temp[cat_temp], drop_first=True)
X_num = X_temp[num_temp]
X_new = pd.concat([X_num, X_cat], axis=1)
new_price_pred = model.predict(X_new)

# Output the predicted price
print(f"Predicted price for the new flat: {new_price_pred[0]:.2f}")
"""