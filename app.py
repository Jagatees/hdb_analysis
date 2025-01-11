import streamlit as st

# Page Title
st.title("HDB Resale Price Prediction")

# Sidebar for Navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select Page", ["Home", "Data Exploration", "Price Prediction"])

if options == "Home":
    st.subheader("Welcome to the HDB Resale Price Prediction Tool")
    st.write("Use this tool to explore HDB resale data and predict prices based on key attributes.")

if options == "Data Exploration":
    st.write("Upload your resale dataset here:")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file:
        st.write("File uploaded successfully! Data will be displayed here.")
    else:
        st.info("No file uploaded yet.")

if options == "Price Prediction":
    st.write("Enter the details below:")
    flat_type = st.selectbox("Flat Type", ["3-room", "4-room", "5-room", "Executive"])
    floor_area = st.number_input("Floor Area (sqm)", min_value=20, max_value=200, step=1)
    lease_remaining = st.slider("Remaining Lease (years)", min_value=1, max_value=99, step=1)
    town = st.selectbox("Town", ["Ang Mo Kio", "Bedok", "Bishan", "Jurong East", "Pasir Ris"])

    # Placeholder for prediction
    if st.button("Predict Price"):
        st.success(f"Predicted Price: ${'500,000'} (Placeholder)")

