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

st.title("💼 Salary Predictor")

# ================================
# INPUTS (RAW FEATURES ONLY)
# ================================

workexp = st.slider("Work Experience", 0, 50, 5)
yearscode = st.slider("Years Coding", 0, 50, 7)
age = st.slider("Age", 15, 75, 30)

country = st.text_input("Country", "India")
devtype = st.text_input("Developer Role", "Developer, full-stack")
edlevel = st.text_input("Education Level", "Bachelor’s degree")
orgsize = st.text_input("Company Size", "20 to 99 employees")

remotework = st.selectbox(
    "Remote Work",
    ["Remote", "Hybrid", "In-person"]
)

employment = st.selectbox(
    "Employment Type",
    ["Employed", "Student", "Freelancer", "Not employed"]
)

# ================================
# PREDICTION
# ================================
if st.button("Predict Salary"):

    # EXACT STRUCTURE MODEL EXPECTS
    input_df = pd.DataFrame({
        "Country": [country],
        "WorkExp": [float(workexp)],
        "YearsCode": [float(yearscode)],
        "DevType": [devtype],
        "EdLevel": [edlevel],
        "OrgSize": [orgsize],
        "Age": [float(age)],
        "RemoteWork": [remotework],
        "Employment": [employment]
    })

    try:
        pred = model.predict(input_df)[0]
        st.success(f"💰 Estimated Salary: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
