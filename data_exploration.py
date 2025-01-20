import folium
import pandas as pd
from geopy.distance import geodesic
import streamlit as st
from streamlit_folium import st_folium


def load_data(uploaded_file):
    """Load the CSV files containing the dataset and bus stops."""
    df = pd.read_csv(uploaded_file)
    bus_stops_df = pd.read_csv("bus_stops_coordinates.csv")
    return df, bus_stops_df


def filter_by_town(df):
    """Filter the dataset based on the selected town."""
    town = st.selectbox("Town", df["town"].unique())
    filtered_df = df[df["town"] == town]
    return filtered_df, town


def get_selected_address(filtered_df):
    """Get the selected address and its latitude/longitude."""
    address = st.selectbox("Address", filtered_df["full_address"].unique())
    selected_row = filtered_df[filtered_df["full_address"] == address].iloc[0]
    selected_lat = selected_row["latitude"]
    selected_lon = selected_row["longitude"]
    return address, selected_lat, selected_lon


def create_map(reduced_df, selected_lat, selected_lon, address):
    """Create a folium map and add markers for the addresses."""
    # center_lat = reduced_df["latitude"].mean()
    # center_lon = reduced_df["longitude"].mean()
    # map_ = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Start from the selected lat, lon
    map_ = folium.Map(location=[selected_lat, selected_lon], zoom_start=15)

    # Add markers for the selected town clinics
    for _, row in reduced_df.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=row.get("full_address", "Address not provided"),
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(map_)

    # Add a red marker for the selected address
    folium.Marker(
        location=[selected_lat, selected_lon],
        popup=f"Selected Address: {address}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(map_)

    return map_


def find_nearby_bus_stops(selected_lat, selected_lon, bus_stops_df, distance_threshold=200):
    """Find bus stops within a given distance (default 200 meters) of the selected address."""
    nearby_bus_stops = []
    for _, bus_stop in bus_stops_df.iterrows():
        bus_lat = bus_stop["Latitude"]
        bus_lon = bus_stop["Longitude"]
        bus_location = (bus_lat, bus_lon)
        selected_location = (selected_lat, selected_lon)

        # Calculate distance between the selected address and each bus stop
        distance = geodesic(selected_location, bus_location).meters
        if distance <= distance_threshold:
            nearby_bus_stops.append(bus_stop)

    return nearby_bus_stops


def add_bus_stop_markers(map_, nearby_bus_stops):
    """Add markers for nearby bus stops on the map."""
    for _, bus_stop in pd.DataFrame(nearby_bus_stops).iterrows():
        folium.Marker(
            location=[bus_stop["Latitude"], bus_stop["Longitude"]],
            popup=f"Bus Stop Code: {bus_stop.get('BusStopCode', 'Code not available')}",
            icon=folium.Icon(color="green", icon="cloud"),
        ).add_to(map_)


def data_exploration(uploaded_file):
    """Main function to handle the data exploration and map display."""
    if uploaded_file:
        df, bus_stops_df = load_data(uploaded_file)

        # Filter data by selected town and address
        filtered_df, town = filter_by_town(df)
        address, selected_lat, selected_lon = get_selected_address(filtered_df)
        # Display the dataset preview
        st.write("Dataset Preview:")
        st.write(df[df["town"] == town])

        if not filtered_df.empty:
            if not pd.api.types.is_numeric_dtype(filtered_df["latitude"]) or not pd.api.types.is_numeric_dtype(filtered_df["longitude"]):
                st.error("Latitude and longitude columns must be numeric.")
            else:
                # Reduce the dataset to the first 100 rows of selected town to prevent performance degrade
                reduced_df = filtered_df.head(100)

                # Create map with markers for addresses
                map_ = create_map(reduced_df, selected_lat, selected_lon, address)

                # Find and add nearby bus stops to the map
                nearby_bus_stops = find_nearby_bus_stops(selected_lat, selected_lon, bus_stops_df)
                add_bus_stop_markers(map_, nearby_bus_stops)

                # Display the map
                st.write("Map showing the addresses based on selected town:")
                st_folium(map_, width=700, height=500)
        else:
            st.warning(f"No data available for the town: {town}")
    else:
        st.info("No file uploaded yet. Please upload a CSV file.")


    # if uploaded_file and bus_stop_uploaded_file:
    #     # Load the uploaded file
    #     df = pd.read_csv(uploaded_file)
    #     bus_stops_df = pd.read_csv(bus_stop_uploaded_file)

    #     # Display the dataset
    #     st.write("Dataset Preview:")
    #     st.write(df.head(100))

    #     # Collect only unique Towns 
    #     town = st.selectbox("Town", df.head(100)["town"].unique())
        
    #     # Filter the DataFrame based on the selected town
    #     filtered_df = df[df["town"] == town]
    #     # Filter address based on town
    #     address = st.selectbox("Address", filtered_df.head(100)["full_address"].unique())
    #     # Find the latitude and longitude for the selected address
    #     selected_row = filtered_df[filtered_df["full_address"] == address].iloc[0]
    #     selected_lat = selected_row["latitude"]
    #     selected_lon = selected_row["longitude"]

    #     if not filtered_df.empty:
    #         # Check if latitude and longitude are numeric
    #         if not pd.api.types.is_numeric_dtype(filtered_df["latitude"]) or not pd.api.types.is_numeric_dtype(filtered_df["longitude"]):
    #             st.error("Latitude and longitude columns must be numeric.")
    #         else:
    #             # Reduce the dataset to the first 100 rows
    #             reduced_df = filtered_df.head(100)

    #             # Initialize a map centered at the average latitude and longitude of the reduced dataset
    #             center_lat = reduced_df["latitude"].mean()
    #             center_lon = reduced_df["longitude"].mean()
    #             map = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    #             # Add markers for the selected town clinics
    #             for _, row in reduced_df.iterrows():
    #                 folium.Marker(
    #                     location=[row["latitude"], row["longitude"]],
    #                     popup=row.get("full_address", "Address not provided"),  # Handle missing full_address
    #                     icon=folium.Icon(color="blue", icon="info-sign"),
    #                 ).add_to(map)

    #             # Add a red marker for the selected address
    #             folium.Marker(
    #                 location=[selected_lat, selected_lon],
    #                 popup=f"Selected Address: {address}",
    #                 icon=folium.Icon(color="red", icon="info-sign"),
    #             ).add_to(map)

    #             # Now, find nearby bus stops (within 100 meters)
    #             nearby_bus_stops = []

    #             for _, bus_stop in bus_stops_df.iterrows():
    #                 bus_lat = bus_stop["Latitude"]
    #                 bus_lon = bus_stop["Longitude"]
    #                 bus_location = (bus_lat, bus_lon)
    #                 selected_location = (selected_lat, selected_lon)
                    
    #                 # Calculate distance between the selected address and each bus stop
    #                 distance = geodesic(selected_location, bus_location).meters
                    
    #                 if distance <= 200:  # Check if bus stop is within 100 meters
    #                     nearby_bus_stops.append(bus_stop)

    #             # Add markers for nearby bus stops (if any)
    #             if nearby_bus_stops:
    #                 for _, bus_stop in pd.DataFrame(nearby_bus_stops).iterrows():
    #                     folium.Marker(
    #                         location=[bus_stop["Latitude"], bus_stop["Longitude"]],
    #                         popup=f"Bus Stop Code: {bus_stop.get('BusStopCode', 'Code not available')}",
    #                         icon=folium.Icon(color="green", icon="cloud"),
    #                     ).add_to(map)

    #             # Display the map in Streamlit
    #             st.write("Map showing the addresses based on selected town:")
    #             st_folium(map, width=700, height=500)
    #     else:
    #         st.warning(f"No data available for the town: {town}")
    # else:
    #     st.info("No file uploaded yet. Please upload a CSV file.")
