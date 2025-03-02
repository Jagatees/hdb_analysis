import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def lease_to_months(lease_str):
    match = re.search(r'(\d+)\s+years(?:\s+(\d+)\s+months)?', lease_str)
    if match:
        years = int(match.group(1))
        months = int(match.group(2)) if match.group(2) else 0
        return years * 12 + months
    return None

# Load dataset
df = pd.read_csv("csv/dataset_for_model.csv")
# ['month','town','flat_type','storey_range','floor_area_sqm','flat_model', 'remaining_lease', 'resale_price','Score']


# Check for the amount of unique values in each independant categorical variable for one hot vector later.
# Not ideal for KNN -> reduces effectivness of distance calculations
# General rule: 10-20 more unique is considered high. 
# Thus town, remaining lease is pretty high. For now, just change remaining lease to numerical manually
columns = ['town','flat_type','storey_range','flat_model', 'remaining_lease']
for col in columns:
    unique_vals = df[col].unique()
    # print(f"Unique values in {col} ({len(unique_vals)}):")
    # print(unique_vals)
    # print()  # Adds an empty line for better readability


# Change the labels in remaining_lease to continuous numbers
df['remaining_lease'] = df['remaining_lease'].apply(lease_to_months)

# Define X and Y
X_temp = df.drop(['resale_price', 'month'], axis=1)
y = df['resale_price']

# Convert texts to numericals
cat_cols = X_temp.select_dtypes(include=['object']).columns
num_cols = X_temp.select_dtypes(exclude=['object']).columns
X_cat = pd.get_dummies(X_temp[cat_cols], drop_first=True)
X_num = X_temp[num_cols]
X = pd.concat([X_num, X_cat], axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
# Use ColumnTransformer to scale only the numerical columns from X_train.
# Because one hot vector columns are already on binary range 0 or 1. To prevent distorting their meaning
ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols)
    ],
    remainder='passthrough'  # This leaves the one-hot columns unchanged
)
X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

# Hyperparameters that i can check
# no. k, distance metric (euclidean), 

# Train KNN model. 5 nearest neighbors
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Predict target values for the normalized test set
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Default Model:")
print(f"Mean Absolute Error: {mae:.2f}, R² Score: {r2:.2f}, MSE Error : {mse:.2f} \n")


# Test new data prediction if needed
# CHANGES MAKE SURE THE VALUES MATCH EXACTLY LIKE THE DATASET*** Bedok etc.
new_data = pd.DataFrame({
    'month': ['2017-01'],
    'town': ['BEDOK'],
    'flat_type': ['3 ROOM'],
    'storey_range': ['04 TO 06'],
    'floor_area_sqm': [70],
    'flat_model': ['New Generation'],
    'remaining_lease': ['61 years 06 months'],
    'resale_price': [280000],
    'Score': [53],
})

# Change the remaining_lease to continuous value like the method 1
new_data['remaining_lease'] = new_data['remaining_lease'].apply(lease_to_months)

X_newTemp = new_data.drop(['resale_price', 'month'], axis=1)
y_new = df['resale_price']

cat_temp = X_newTemp.select_dtypes(include=['object']).columns
num_temp = X_newTemp.select_dtypes(exclude=['object']).columns
# Change to drop_first=false, because true is removing every categorical column which results to inaccuracy
X_cat_new = pd.get_dummies(X_newTemp[cat_temp], drop_first=False)
X_num_new = X_newTemp[num_temp]
X_new = pd.concat([X_num_new, X_cat_new], axis=1)


# print(X_new)

# Ensures the new data has the same columns as X_train
X_new = X_new.reindex(columns=X_train.columns, fill_value=0)

# pd.set_option('display.max_columns', None)
# print(X_new)

# Apply the previously fitted ColumnTransformer to scale the continuous features
X_new_scaled = ct.transform(X_new)

# Now you can pass X_new_scaled to your trained model for prediction:
new_price_pred = model.predict(X_new_scaled)
print(f"Predicted price for the new flat: {new_price_pred[0]:.2f}\n")


# Fine tuning
pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('knn', KNeighborsRegressor())  
])

# Hyperparameter grid
param_grid = {
    'knn__n_neighbors': list(range(1, 21)),  # K-values from 1 to 20
    'knn__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'knn__weights': ['uniform', 'distance']
}

# Grid Search with Cross Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Show best params
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Train Best Model
best_knn = grid_search.best_estimator_
y_pred_tuned = best_knn.predict(X_test_scaled)

# Evaluate Tuned Model
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"Tuned Model:")
print(f"Mean Absolute Error: {mae_tuned:.2f}, R² Score: {r2_tuned:.2f}, MSE Error : {mse_tuned:.2f} \n")
