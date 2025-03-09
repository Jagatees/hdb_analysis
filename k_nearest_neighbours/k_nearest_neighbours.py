import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Load dataset
df = pd.read_csv("sampled_hdb_data.csv")

# Define X and Y
# X_temp = df.drop(['resale_price', 'month', 'town', 'storey_range'], axis=1)
# X_temp = df.drop(['resale_price', 'town', 'storey_range', 'latitude', 'longitude'], axis=1)
# X_temp = df.drop(['resale_price', 'town', 'storey_range', 'block', 'street_name', 'full_address', 'latitude', 'longitude'], axis=1)
X_temp = df[['year', 'flat_type', 'floor_area_sqm', 'remaining_lease', 'Score', 'storey_range_numeric', 'region']]
y = df['resale_price']

# Extract numerical and categorical values
cat_cols = X_temp.select_dtypes(include=['object']).columns
num_cols = X_temp.select_dtypes(exclude=['object']).columns
X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]
X = pd.concat([X_num, X_cat], axis=1)

# Compute correlation matrix
corr_matrix = X.corr()

# Find highly correlated features (threshold > 0.90)
high_corr_features = np.where(np.abs(corr_matrix) > 0.90)

# Extract feature pairs (excluding diagonal elements)
redundant_pairs = [(corr_matrix.index[i], corr_matrix.columns[j]) 
                   for i, j in zip(*high_corr_features) if i != j]

print("Highly Correlated Feature Pairs (>|0.90|):", redundant_pairs)

# Plot correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
# Save the figure correlation heatmap
plt.savefig("knn/correlation_heatmap.png", dpi=300, bbox_inches='tight')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Select relevant features only so my feature importance does not have too many columns to calculate
top_features = X_train.corrwith(y_train).abs().sort_values(ascending=False).index[:10]
print(top_features)

# Normalisation
ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols)
    ],
    remainder='passthrough'  # This leaves the one-hot columns unchanged
)
X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Train KNN model. 5 nearest neighbors
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Predict target values for the normalized test set
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Default Model:")
print(f"MSE: {mse:.2f}, RMSE {rmse:.2f}, MAE {mae:.2f}, R²: {r2:.2f}",)

# Feature Importance
# Compute permutation importance
perm_importance = permutation_importance(model, X_test_scaled, y_test, scoring='neg_mean_squared_error', n_repeats=1)

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': perm_importance.importances_mean})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df["Normalized Importance"] = feature_importance_df["Importance"] / feature_importance_df["Importance"].max()
feature_importance_df.to_csv("knn/feature_importance.csv", index=False)

# # Plot feature importance
# plt.figure(figsize=(12, 6))
# sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
# plt.xlabel("Feature Importance (Permutation)")
# plt.ylabel("Features")
# plt.title("Feature Importance for KNN")
# plt.grid(axis="x", linestyle="--", alpha=0.7)

# # Save the plot
# plt.savefig("knn/feature_importance_plot.pn", dpi=300, bbox_inches="tight")
# plt.show()
