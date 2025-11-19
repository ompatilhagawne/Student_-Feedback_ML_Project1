import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Local File Paths (GitHub)
# -----------------------------
MODEL_PATH = "Student_model.pkl"
DATA_PATH = "Employee_clean_Data.csv"

# -----------------------------
# Load Model and Data
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Make sure Student_model.pkl is uploaded in the repo.")
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Data file not found. Make sure Employee_clean_Data.csv is uploaded.")
        return None
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_data()

st.title("Student Prediction App")

if model is None or data is None:
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# Dynamic Inputs
# -----------------------------
st.subheader("Enter Input Values")

input_cols = [col for col in data.columns if data[col].dtype != 'object']
inputs = {}

for col in input_cols:
    inputs[col] = st.number_input(
        f"Enter {col}", 
        value=float(data[col].mean())
    )

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    try:
        pred = model.predict(df)[0]
        st.success(f"Prediction: {pred}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
