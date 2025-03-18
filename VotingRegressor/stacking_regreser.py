import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a regression model and print key metrics.
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print("Performance Metrics:")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return r2, mse, rmse, mae

def main():
    # -----------------------------
    # Data Loading
    # -----------------------------
    # Load training dataset (2017-2023) and 2024 data for prediction
    train_data_path = "../csv/2017 - 2023.csv"
    prediction_data_path = "../csv/2024.csv"
    
    df = pd.read_csv(train_data_path)
    df_2024 = pd.read_csv(prediction_data_path)
    
    # -----------------------------
    # Feature Definition
    # -----------------------------
    features = [
        'year',
        'month',
        'floor_area_sqm',
        'storey_range_numeric',
        'remaining_lease',
        'score'
    ]
    categorical_features = [
        'town',
        'flat_type',
        'flat_model',
        'region'
    ]
    target = 'resale_price'
    
    # -----------------------------
    # Preprocessing Pipeline
    # -----------------------------
    # Standardize numerical features and one-hot encode categorical features.
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    
    # -----------------------------
    # Model Pipelines: Two XGBoost Variants
    # -----------------------------
    # First XGBoost pipeline with tuned parameters
    pipeline_xgb1 = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        ))
    ])
    
    # Second XGBoost pipeline with different tuned parameters
    pipeline_xgb2 = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9
        ))
    ])
    
    # -----------------------------
    # Stacking Regressor Setup
    # -----------------------------
    # Combine both XGBoost models in a stacking ensemble with a Ridge meta-model.
    estimators = [
        ('xgb1', pipeline_xgb1),
        ('xgb2', pipeline_xgb2)
    ]
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    # -----------------------------
    # Train/Test Split and Model Fitting
    # -----------------------------
    X = df[features + categorical_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting the stacking regressor...")
    stacking_regressor.fit(X_train, y_train)
    
    # -----------------------------
    # Model Evaluation on Test Set
    # -----------------------------
    print("\nStacking Regressor Performance on Test Set:")
    y_pred = stacking_regressor.predict(X_test)
    evaluate_model(y_test, y_pred)
    
    # -----------------------------
    # Prediction on 2024 Data
    # -----------------------------
    X_2024 = df_2024[features + categorical_features]
    df_2024['predicted_resale_price'] = stacking_regressor.predict(X_2024)
    
    # Calculate overall prediction loss percentage on 2024 data
    total_loss = np.sum(np.abs(df_2024['resale_price'] - df_2024['predicted_resale_price']))
    total_actual = np.sum(df_2024['resale_price'])
    loss_percentage = (total_loss / total_actual) * 100
    print(f"\nStacking Regressor Overall Prediction Loss Percentage: {loss_percentage:.2f}%")
    
    # Save predictions to CSV
    output_path = "../csv_predicated_model/stacking_regressor.csv"
    df_2024.to_csv(output_path, index=False)
    print(f"Predicted resale prices for 2024 saved to {output_path}")

    



if __name__ == '__main__':
    main()



