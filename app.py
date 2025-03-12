import streamlit as st
import pandas as pd
import folium
import os
from streamlit_folium import st_folium

# Page Title
st.title("HDB Resale Price Prediction with Map Visualization")

# Sidebar for Navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Select Page",
    ["Home", "Map Visualization"]
)

# Folder where CSVs are stored
csv_folder = "csv_predicated_model"

# ------------------
# HOME PAGE
# ------------------
if options == "Home":
    st.subheader("Welcome to the HDB Resale Price Prediction Tool")
    st.write("This tool allows you to visualize 2024 HDB resale prices and predicted prices on a map.")

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
