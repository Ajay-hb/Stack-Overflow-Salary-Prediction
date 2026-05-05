import streamlit as st
import joblib
import pandas as pd

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    return joblib.load("salary_model_compressed.joblib")

model = load_model()

# ================================
# UI
# ================================
st.title("💼 Salary Predictor")

# Numeric
exp = st.slider("Work Experience", 0, 50, 5)
code = st.slider("Years Coding", 0, 50, 7)
age = st.slider("Age", 15, 75, 30)

# RAW categorical inputs
country = st.text_input("Country", "India")
devtype = st.text_input("Developer Role", "Developer, full-stack")
ed = st.text_input("Education", "Bachelor’s degree")
org = st.text_input("Company Size", "20 to 99 employees")
remote = st.selectbox("Remote Work", ["Remote", "Hybrid", "In-person"])
emp = st.selectbox("Employment", ["Employed", "Student", "Freelancer"])

# ================================
# PREDICTION
# ================================
if st.button("Predict Salary"):

    # 👉 RAW INPUT ONLY (NO ONE-HOT)
    input_df = pd.DataFrame({
        "Country": [country],
        "WorkExp": [float(exp)],
        "YearsCode": [float(code)],
        "DevType": [devtype],
        "EdLevel": [ed],
        "OrgSize": [org],
        "Age": [float(age)],
        "RemoteWork": [remote],
        "Employment": [emp]
    })

    # Predict
    pred = model.predict(input_df)[0]

    st.success(f"💰 Estimated Salary: ${pred:,.2f} USD")
