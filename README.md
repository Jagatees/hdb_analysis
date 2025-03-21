# 🏠 HDB Resale Price Prediction
This project leverages machine learning to predict HDB resale prices in Singapore. Built with Streamlit, it provides an interactive interface where users can input various parameters through a dropdown menu and receive real-time price predictions.

# 📌 Introduction
This report outlines the development of a regression model designed to predict fair resale prices for public housing flats in Singapore. The model is trained using historical HDB resale transaction data from 2017 to 2023, allowing it to predict 2024 resale prices.

Since we already have actual resale prices for 2024, we can evaluate our model's accuracy by comparing predicted prices against the real 2024 transactions. This enables us to assess how well different models perform in forecasting resale values and identify the most reliable approach for future predictions.

By integrating machine learning techniques, this project aims to assist homeowners, buyers, and policymakers in making data-driven decisions regarding HDB resale transactions.

## Dataset

### HDB resale price prediction:
- https://data.gov.sg/dataset/resale-flat-prices

### Coordinates of Amenities:

#### Shopping Mall:  
- https://www.kaggle.com/datasets/karthikgangula/shopping-mall-coordinates?resource=download  

#### School:
- https://data.gov.sg/datasets/d_688b934f82c1059ed0a6993d2a829089/view  

#### Bus Stop:  
- https://www.kaggle.com/datasets/gowthamvarma/singapore-bus-data-land-transport-authority/data  

#### MRT Station:  
- https://www.kaggle.com/datasets/shengjunlim/singapore-mrt-lrt-stations-with-coordinates?resource=download
- https://www.kaggle.com/datasets/yxlee245/singapore-train-station-coordinates  

#### Polyclinics:  
- https://www.arcgis.com/home/item.html?id=f3c1113a2c554716a2a1a0d3632dd9ed&sublayer=0#data  

#### Hospital:  
- https://www.pacificprime.sg/resources/singapore-hospital-list/  

## 🚀 Features

- **🔹 Interactive UI:** A user-friendly interface powered by **Streamlit**.
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


## 📊 Data Attributes

### **Features (Numerical)**
These features provide **quantitative** insights into resale transactions:

| No. | Name                 | Type          | Description |
|----|----------------------|--------------|-------------|
| 1  | **Year**             | Numeric (YYYY) | Extracted from the transaction date to identify **seasonal trends**. |
| 2  | **Month**            | Numeric (MM)   | Helps capture **month-wise price variations**. |
| 3  | **Floor Area (sqm)** | Numeric (sqm)  | Represents the **size of the flat** in square meters. |
| 4  | **Storey Range Numeric** | Numeric  | Converted from categorical **storey range** into a numeric format for better model processing. |
| 5  | **Remaining Lease**  | Numeric (Years) | Originally categorical (YY-MM format), converted into **years** to represent lease duration. |
| 6  | **Score**            | Numeric        | A derived feature that incorporates **location-based desirability and pricing trends**. |

### **Categorical Features**
These categorical variables provide **contextual information** about the flats:

| No. | Name        | Type              | Description |
|----|------------|------------------|-------------|
| 1  | **Town**   | Categorical Text  | Represents the **geographical location** of the flat in Singapore. |
| 2  | **Flat Type** | Categorical Text  | Specifies the **type of unit** (e.g., **3-room, 4-room, 5-room, Executive**). |
| 3  | **Flat Model** | Categorical Text  | Indicates the **design and layout** of the unit (e.g., **Model A, Improved, Maisonette**). |
| 4  | **Region**  | Categorical Text  | Groups towns into **broader regions** (e.g., **Central, North, East, West, Northeast**) for location-based trends. |

### **Data Information**
Numerical features 1-5 and Categorical Features 1-4 were derived from the dataset found in [HDB resale price prediction](https://data.gov.sg/dataset/resale-flat-prices)  

Numerical features 6 (Score) is derived from adding weights to each of the amenities listed in the dataset to generate a score for each of the HDB listed in the [dataset](https://data.gov.sg/dataset/resale-flat-prices)  

---

## 🔍 **Feature Selection**

To build an **accurate prediction model for HDB resale prices**, we leverage a combination of **numerical** and **categorical** features.

### **Selected Features (Numerical)**
- **Year, Month** → Extracted from transaction data to identify **seasonal trends**.
- **Floor Area (sqm)** → Represents the **size of the flat** in square meters.
- **Storey Range Numeric** → Converted from categorical **storey range** to numerical values for better model processing.
- **Remaining Lease** → Originally categorical (YY-MM format), converted into **numerical years** to represent the **remaining lease duration**.
- **Score** → A **derived feature** incorporating **location-based desirability and pricing trends**.

### **Selected Categorical Features**
- **Town** → Represents the **geographical location** of the flat in Singapore.
- **Flat Type** → Specifies the **unit type** (e.g., **3-room, 4-room, 5-room, Executive**).
- **Flat Model** → Defines the **layout and structure** of the flat (e.g., **Model A, Improved, Maisonette**).
- **Region** → Groups **towns into broader regions** (e.g., **Central, North, East, West, Northeast**) for location-based trends.

By using a combination of **numerical** and **categorical** features, our model captures both **quantifiable** and **qualitative** aspects of HDB resale pricing, ensuring better prediction accuracy.

---

### ✅ **Key Improvements in Data Processing**
- 📅 **Month** column split into **Year** and **Month** to detect seasonal trends.
- 🔢 **Storey Range** converted into **numerical values** for better model performance.
- ⏳ **Remaining Lease** transformed from **categorical format** into numerical **years** for improved accuracy.
## 🛠 **Splitting Data for Train, Test & Validation**

### 🌐 **Data Splitting Strategy**
To ensure a robust and realistic model evaluation, we use a **time-based train-test split** instead of a random split:

- **Training Set (2017 - 2023)**: Used to train the model on historical trends.
- **Validation & Testing Set (20% of 2017 - 2023 data)**: Used to evaluate model performance before making predictions.
- **Prediction Set (2024)**: The model predicts **2024 resale prices**, which we compare against actual 2024 prices to measure accuracy and error rates.

### 💪 **Why This Approach?**
Unlike random train-test splits, our **time-based approach** prevents **data leakage from future transactions**, making the prediction process more realistic.
## 🏆 Best Model Performance

We evaluated multiple models, and here are the 11 models ranked accordingly by loss percentage:

| Rank | Model                                 | R² Score | RMSE       | MSE               | MAE        | Loss Percentage % |
|---|---------------------------------------|----------|------------|-------------------|------------|-------------------|
| 1️ | **Stacking Regressor (XG + XGBoost)** | 0.9472   | 38250.11   | 1,463,070,655.89  | 27,024.72  | 8.06              |
| 2️ | **XGBoost with GridSearch**     | 0.9498   | 37297.22   | 1,391,082,733.51  | 26,133.19  | 8.20              |
| 3️ | **XGBoost (Standard)**                | 0.9460   | 38710.85   | 1,498,530,194.50  | 27,413.72  | 8.39              |
| 4️ | **Voting Regressor**                  | 0.9420   | 40094.02   | 1,607,530,584.93  | 27,576.23  | 8.53              |
| 5 | **RandomForestRegressor (Base)**      | 0.9290   | 44355.55   | 1,967,414,894.78  | 29,697.01  | 9.09              |
| 6️ | **Linear Regression**                 | 0.8626   | 61724.76   | 3,809,946,402.46  | 48,092.03  | 9.76              |
| 7️ | **Decision Tree with GridSearch**     | 0.8810   | 57435.65   | 3,298,853,944.70  | 38,937.18  | 10.17             |
| 8 | **RandomForest with GridSearch**      | 0.9312   | 43671.19   | 1,907,172,613.87  | 29,655.97  | 10.32             |
| 9️ | **Decision Tree**                     | 0.8701   | 60009.04   | 3,601,085,294.17  | 40,872.11  | 10.39             |
| 10 | **KNeighborsRegressor (Base)**        | 0.8831   | 56945.11   | 3,242,746,014.74  | 38,346.69  | 10.62             |
| 11 | **KNN with GridSearch**               | 0.9058   | 51121.55   | 2,613,413,116.25  | 34,805.14  | 11.44             |
