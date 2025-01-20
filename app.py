import streamlit as st
from data_exploration import data_exploration

# Page Title
st.title("HDB Resale Price Prediction with Map Visualization")

# Sidebar for Navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select Page", ["Home", "Data Exploration", "Price Prediction"])

if options == "Home":
    st.subheader("Welcome to the HDB Resale Price Prediction Tool")
    st.write("Use this tool to explore HDB resale data and predict prices based on key attributes.")

if options == "Data Exploration":
    st.subheader("Data Exploration")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    data_exploration(uploaded_file)

if options == "Price Prediction":
    st.subheader("Price Prediction")
    flat_type = st.selectbox("Flat Type", ["3-room", "4-room", "5-room", "Executive"])
    floor_area = st.number_input("Floor Area (sqm)", min_value=20, max_value=200, step=1)
    lease_remaining = st.slider("Remaining Lease (years)", min_value=1, max_value=99, step=1)
    town = st.selectbox("Town", ["Ang Mo Kio", "Bedok", "Bishan", "Jurong East", "Pasir Ris"])

    # Placeholder for prediction
    if st.button("Predict Price"):
        st.success(f"Predicted Price: ${'500,000'} (Placeholder)")
