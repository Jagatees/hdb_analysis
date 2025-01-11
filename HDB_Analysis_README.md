# HDB Analysis

This project is designed to analyze HDB resale prices using Streamlit and Python.

## Setup Instructions
Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone <repository-url>
cd HDB_ANALYSIS
```

### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
```
Activate the virtual environment:
- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- On **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Required Packages
Install the dependencies listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
To launch the Streamlit app, use the following command:
```bash
streamlit run app.py
```
This will start a local web server, and you can open the app in your browser at:
```
http://localhost:8501
```

### 5. Deactivate the Virtual Environment
When you are done, deactivate the virtual environment:
```bash
deactivate
```

---

## Notes
- Ensure you have Python 3.8 or later installed.
- If any packages are missing, update the `requirements.txt` file accordingly and run:
  ```bash
  pip freeze > requirements.txt
  ```
- For additional troubleshooting, refer to the Streamlit [documentation](https://docs.streamlit.io/).

---

Feel free to reach out if you encounter any issues!
