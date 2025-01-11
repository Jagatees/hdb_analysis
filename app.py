import streamlit as st

# Page Title
st.title("HDB Resale Price Prediction")

# Sidebar for Navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select Page", ["Home", "Data Exploration", "Price Prediction"])

if options == "Home":
    st.subheader("Welcome to the HDB Resale Price Prediction Tool")
    st.write("Use this tool to explore HDB resale data and predict prices based on key attributes.")

elif options == "Data Exploration":
    st.subheader("Explore Resale Data")
    st.write("Visualize and analyze the trends in HDB resale data.")

elif options == "Price Prediction":
    st.subheader("Predict Resale Prices")
    st.write("Enter the details below to get a resale price prediction.")
