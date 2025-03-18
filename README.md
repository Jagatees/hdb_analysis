# 🏠 HDB Resale Price Prediction

This project leverages **machine learning** to predict HDB resale prices. Built with **Streamlit**, it provides an interactive interface where users can input parameters through a dropdown menu and receive real-time price predictions.
## 🚀 Features

- **🔹 Interactive UI:** A user-friendly interface powered by **Streamlit**.
- **📊 Real-Time Predictions:** Instant resale price predictions based on user-selected features.
- **📂 Data-Driven Model:** Trained on historical HDB resale transaction data.
- **⚡ Optimized Performance:** Hyperparameter tuning to improve prediction accuracy.
## 🛠 Requirements

- Python **3.x**
- Required dependencies listed in **`requirements.txt`**
    
## Deployment

To run Project

```bash
cd <project_directory>
pip install -r requirements.txt
python3 app.py
```

## 🏆 Best Model Performance

We evaluated multiple models, and here are the top three based on performance metrics:

| Model                               | R² Score | RMSE (SGD) | MSE (SGD²) | MAE (SGD) | Prediction Loss % |
|--------------------------------------|---------|------------|------------|------------|------------------|
| **XGBoost with GridSearch** (Best)   | **0.9498** | **37,297.22** | **1,391,082,733.51** | **26,133.19** | **8.20%** |
| **Stacking Regressor (XG + XGBoost)** | 0.9472 | 38,250.11 | 1,463,070,655.89 | 27,024.72 | 8.06% |
| **XGBoost (Standard)**               | 0.9460 | 38,710.85 | 1,498,530,194.50 | 27,413.72 | 8.39% |
