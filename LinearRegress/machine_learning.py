import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

town_to_region = {
    "ANG MO KIO": "North-East", "BEDOK": "East", "BISHAN": "Central", "BUKIT BATOK": "West",
    "BUKIT MERAH": "Central", "BUKIT PANJANG": "West", "BUKIT TIMAH": "Central", "CENTRAL AREA": "Central",
    "CHOA CHU KANG": "West", "CLEMENTI": "West", "GEYLANG": "East", "HOUGANG": "North-East",
    "JURONG EAST": "West", "JURONG WEST": "West", "KALLANG/WHAMPOA": "Central", "MARINE PARADE": "East",
    "PASIR RIS": "East", "PUNGGOL": "North-East", "QUEENSTOWN": "Central", "SEMBAWANG": "North",
    "SENGKANG": "North-East", "SERANGOON": "North-East", "TAMPINES": "East", "TOA PAYOH": "Central",
    "WOODLANDS": "North", "YISHUN": "North"
}

# Load dataset
df = pd.read_csv('sampled_hdb_data (2).csv')

# Encode categorical variable 'storey_range'
le = LabelEncoder()
df['storey_range_encoded'] = le.fit_transform(df['storey_range'])

df["region"] = df["town"].map(town_to_region)

print(df["region"].value_counts())
print(df["region"].shape[0])

# Feature selection: Drop non-relevant target variables
X_temp = df.drop(columns=['resale_price', 'remaining_lease'], errors='ignore')

# Identify categorical & numerical columns
cat_cols = X_temp.select_dtypes(include=['object']).columns
num_cols = X_temp.select_dtypes(exclude=['object']).columns

# Encoding Strategy: Label Encode High-Cardinality Features, One-Hot Encode Others
for col in cat_cols:
    if df[col].nunique() > 50:  # Threshold for high-cardinality categorical variables
        le = LabelEncoder()
        X_temp[col] = le.fit_transform(X_temp[col])
    else:
        X_temp = pd.get_dummies(X_temp, columns=[col], drop_first=True, dtype=np.float32)

# Combine numerical & encoded categorical columns
X_num = X_temp[num_cols].astype(np.float32)  # Convert to float32 to reduce memory
X = pd.concat([X_num, X_temp.drop(columns=num_cols)], axis=1)

# Standardize numerical features
scaler = StandardScaler()
X[X_num.columns] = scaler.fit_transform(X[X_num.columns])

# Feature Selection: Drop highly correlated features (>0.95 correlation)
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X.drop(columns=to_drop, inplace=True)

# Dimensionality Reduction using PCA
n_components = min(X.shape[0], X.shape[1])  # Get the max allowed components
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X)

# Define target variable
y = df['resale_price'].astype(np.float32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

# Train Linear Regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predictions
y_pred = reg_model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

# Visualization
sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"})
plt.xlabel("Actual Resale Price")
plt.ylabel("Predicted Resale Price")
plt.title("Linear Regression Predictions vs Actual")
plt.show()
