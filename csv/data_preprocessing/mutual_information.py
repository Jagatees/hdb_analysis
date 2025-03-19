import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

file_path_2017_2023 = "../2017 - 2023.csv"
df = pd.read_csv(file_path_2017_2023)

# Define features and target variable
features = [
    'year',
    'month',
    'floor_area_sqm',
    'storey_range_numeric',
    'remaining_lease',
    'score',
    'lease_commence_date',
    'latitude',
    'longitude',
]
categorical_features = [
    'town',
    'flat_type',
    'flat_model',
    'region',
    'street_name',
    'storey_range',
    'block'
]
target = 'resale_price'

# Prepare independent (X) and dependent (y) variables
X = df[features + categorical_features]

# Apply Standardisation (Z-score scaling) to numerical features
scaler = StandardScaler()
X[features] = scaler.fit_transform(X[features]) 

# Define target variable
y = df[target]

# One-hot encode categorical features for MI calculation
X_encoded = pd.get_dummies(X, drop_first=True)

# Compute Mutual Information Scores
mi_scores = mutual_info_regression(X_encoded, y)

# Store MI scores in a DataFrame
mi_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Mutual Information Score": mi_scores
}).sort_values(by="Mutual Information Score", ascending=False)

# # Save to CSV for further analysis
mi_df.to_csv("mutual_information_scores.csv", index=False)
print("Mutual Information scores saved to 'mutual_information_scores.csv'")

# Plot the top 20 features with highest Mutual Information scores
top_n = 20  # Number of features to display
top_mi_df = mi_df.head(top_n)

plt.figure(figsize=(20, 6))
plt.barh(top_mi_df["Feature"], top_mi_df["Mutual Information Score"], color="skyblue")
plt.xlabel("Mutual Information Score")
plt.ylabel("Feature")
plt.title("Top 20 Features by Mutual Information Score")
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.show()

