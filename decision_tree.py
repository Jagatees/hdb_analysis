import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("csv/dataset_for_model.csv")
# ['month','town','flat_type','storey_range','floor_area_sqm','flat_model', 'remaining_lease', 'resale_price','Score']

# Define X and Y
X_temp = df.drop(['resale_price', 'month'], axis=1)
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
model = DecisionTreeRegressor(max_depth=25, min_samples_leaf=1, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}, R² Score: {r2:.2f}")

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