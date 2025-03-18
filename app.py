import streamlit as st
import pandas as pd
import folium
import os
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

# Page Title
st.title("HDB Resale Price Prediction & Analysis")

# Sidebar for Navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Select Page",
    ["Home", "Map Visualization", "Model Comparison"]
)

# Folder where CSVs are stored
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

    # 4. Check if required columns exist
    required_columns = {'latitude', 'longitude', 'town', 'resale_price', 'predicted_resale_price'}
    missing_columns = required_columns - set(df_map.columns)
    
    if missing_columns:
        st.error(f"Missing columns in CSV: {missing_columns}")
        st.stop()

    # 5. Show preview of selected dataset
    st.write(f"Preview of **{selected_csv}** data:")
    st.dataframe(df_map.head())

    # 6. Create a Folium map centered on Singapore
    folium_map = folium.Map(location=[1.3521, 103.8198], zoom_start=12)

    # 7. Add markers with actual and predicted resale prices
    for _, row in df_map.iterrows():
        popup_info = (
            f"<b>Town:</b> {row['town']}<br>"
            f"<b>Actual Resale Price:</b> ${row['resale_price']:,.2f}<br>"
            f"<b>Predicted Resale Price:</b> ${row['predicted_resale_price']:,.2f}"
        )
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_info, max_width=300),
            icon=folium.Icon(icon='home', prefix='fa', icon_color='white', color='blue')
        ).add_to(folium_map)

    # 8. Display the map in Streamlit
    st_folium(folium_map, width=700, height=500)

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

    # Explanation of the metrics
    st.write("### 📖 Understanding the Metrics")
    
    st.markdown("""
    - **🔵 R² Score (Coefficient of Determination)**: Measures how well the model explains the variance in resale prices.  
      - **Closer to 1** means a better model fit.
    - **🔴 RMSE (Root Mean Squared Error)**: Measures the average difference between predicted and actual prices.  
      - **Lower values are better**.
    - **🟢 MSE (Mean Squared Error)**: Similar to RMSE but squared, penalizing large errors more.  
      - **Lower values are better**.
    - **🟠 MAE (Mean Absolute Error)**: Measures the average absolute error between actual and predicted prices.  
      - **Lower values are better**.
    - **🟣 Prediction Loss %**: Compares total absolute error to actual resale prices, showing **overall model accuracy**.  
      - **Lower values are better**.
    """)

    # Plot Model Performance Comparison
    st.write("### 📊 Model Performance Comparison")

    # Select metrics to visualize
    metric_options = ["R² Score", "RMSE", "MSE", "MAE", "Prediction Loss %"]
    selected_metric = st.selectbox("Select a metric to compare:", metric_options)

    # Plot selected metric as a bar chart with improved layout
    fig, ax = plt.subplots(figsize=(12, 6))
    df_performance.plot(x="Model", y=selected_metric, kind="bar", ax=ax, color='skyblue', legend=False)

    plt.title(f"{selected_metric} Comparison Across Models", fontsize=14, fontweight='bold')
    plt.ylabel(selected_metric, fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=30, ha="right", fontsize=10)  # Rotate and align x-axis labels properly
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
