import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# Page Title
st.title("HDB Resale Price Prediction & Analysis")

# Sidebar for Navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Select Page",
    ["Home", "Map Visualization", "Model Comparison", "Scatter Plot Comparison"]
)

# Folder where predicted CSVs are stored
csv_folder = "csv_predicated_model"

# ------------------
# HOME PAGE
# ------------------
if options == "Home":
    st.subheader("Welcome to the HDB Resale Price Prediction Tool")
    st.write("This tool allows you to visualize 2024 HDB resale prices and predicted prices on a map, as well as compare different machine learning models used for prediction.")

# ------------------
# MAP VISUALIZATION PAGE
# ------------------
elif options == "Map Visualization":
    st.subheader("Map Visualization")

    # 1. Get list of CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    if not csv_files:
        st.error("No CSV files found in the directory.")
        st.stop()

    # 2. Dropdown to select a file
    selected_csv = st.sidebar.selectbox("Select a dataset", csv_files)

    # 3. Load selected CSV
    csv_path = os.path.join(csv_folder, selected_csv)
    try:
        df_map = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Could not find the file: {csv_path}")
        st.stop()

    # 4. Show preview of selected dataset
    st.write(f"Preview of **{selected_csv}** data:")
    st.dataframe(df_map.head())

# ------------------
# MODEL COMPARISON PAGE
# ------------------
elif options == "Model Comparison":
    st.subheader("Model Comparison")

    # Path to the model performance CSV
    performance_csv = "model_performance.csv"

    # Check if the file exists
    if not os.path.exists(performance_csv):
        st.error("Model performance file not found.")
        st.stop()

    # Load model performance data
    df_performance = pd.read_csv(performance_csv)

    # Show raw performance data
    st.write("Performance metrics of different models:")
    st.dataframe(df_performance)

    # Select metrics to visualize
    metric_options = ["R² Score", "RMSE", "MSE", "MAE", "Prediction Loss %"]
    selected_metric = st.selectbox("Select a metric to compare:", metric_options)

    # Plot selected metric as a bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    df_performance.plot(x="Model", y=selected_metric, kind="bar", ax=ax, color='skyblue', legend=False)

    plt.title(f"{selected_metric} Comparison Across Models", fontsize=14, fontweight='bold')
    plt.ylabel(selected_metric, fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the chart as an image
    chart_path = "model_performance_chart.png"
    plt.savefig(chart_path, format="png")

    # Show the chart in Streamlit
    st.pyplot(fig)

    # Provide a download button for the chart
    with open(chart_path, "rb") as file:
        st.download_button(
            label="📥 Download Chart",
            data=file,
            file_name="model_performance_chart.png",
            mime="image/png"
        )

# ------------------
# SCATTER PLOT COMPARISON PAGE
# ------------------
elif options == "Scatter Plot Comparison":
    st.subheader("Model Comparison: Actual vs. Predicted Prices")

    # 1. Get list of prediction CSV files
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    if not csv_files:
        st.error("No prediction CSV files found in the directory.")
        st.stop()

    # 2. Dropdown to select a model's prediction dataset
    selected_csv = st.sidebar.selectbox("Select a model's prediction file:", csv_files)

    # 3. Load selected model's prediction CSV
    csv_path = os.path.join(csv_folder, selected_csv)
    try:
        df_predicted = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Could not find the file: {csv_path}")
        st.stop()

    # 4. Check if required columns exist
    required_columns = {'resale_price', 'predicted_resale_price'}
    missing_columns = required_columns - set(df_predicted.columns)

    if missing_columns:
        st.error(f"Missing columns in CSV: {missing_columns}")
        st.stop()

    # 5. Generate Scatter Plot for Actual vs. Predicted Prices
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_predicted['resale_price'], df_predicted['predicted_resale_price'], 
               color='blue', alpha=0.5, label="Predicted vs. Actual")

    # Add a reference line (Perfect Prediction)
    min_price = min(df_predicted['resale_price'].min(), df_predicted['predicted_resale_price'].min())
    max_price = max(df_predicted['resale_price'].max(), df_predicted['predicted_resale_price'].max())
    ax.plot([min_price, max_price], [min_price, max_price], color="red", linestyle="--", label="Perfect Prediction")

    # Formatting the plot
    ax.set_title(f"Actual vs. Predicted Prices: {selected_csv}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Resale Price (SGD)", fontsize=12)
    ax.set_ylabel("Predicted Resale Price (SGD)", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Save the plot as an image
    scatter_path = f"{selected_csv}_scatter.png"
    plt.savefig(scatter_path, format="png")

    # Show the scatter plot in Streamlit
    st.pyplot(fig)

    # Provide a download button for the chart
    with open(scatter_path, "rb") as file:
        st.download_button(
            label="📥 Download Scatter Plot",
            data=file,
            file_name=f"{selected_csv}_scatter.png",
            mime="image/png"
        )
