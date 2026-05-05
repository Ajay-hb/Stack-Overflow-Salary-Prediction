import streamlit as st
import joblib
import pandas as pd
import re

# ================================
# LOAD MODEL + COLUMNS
# ================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("salary_model_compressed.joblib")
    columns = joblib.load("model_columns.joblib")
    return model, columns

model, MODEL_COLUMNS = load_artifacts()

# ================================
# CLEAN FUNCTION
# ================================
def clean_col_names(df):
    df.columns = [re.sub(r"[^\w]", "", col) for col in df.columns]
    return df

# ================================
# UI
# ================================
st.title("💼 Advanced Salary Predictor")

# Numeric
exp = st.slider("Work Experience", 0, 50, 5)
code = st.slider("Years of Coding (Pro)", 0, 50, 7)
age = st.number_input("Age", value=25.0)

# Extract categories dynamically
def extract(prefix):
    return sorted([c.replace(prefix, "") for c in MODEL_COLUMNS if c.startswith(prefix)])

countries = extract("Country_")
ed_levels = extract("EdLevel_")
remote_opts = extract("RemoteWork_")
org_sizes = extract("OrgSize_")
dev_types = extract("DevType_")

selected_country = st.selectbox("Country", countries)
selected_ed = st.selectbox("Education", ed_levels)
selected_remote = st.selectbox("Remote Work", remote_opts)
selected_org = st.selectbox("Company Size", org_sizes)
selected_dev = st.selectbox("Developer Type", dev_types)

# ================================
# SKILLS (MULTI-SELECT)
# ================================
# Detect skill columns (everything not numeric or categorical)
non_skill_prefixes = ("Country_", "EdLevel_", "RemoteWork_", "OrgSize_", "DevType_")
skill_columns = [c for c in MODEL_COLUMNS if not c.startswith(non_skill_prefixes)
                 and c not in ["WorkExp", "YearsCodePro", "Age"]]

selected_skills = st.multiselect("Select Your Skills", skill_columns)

# ================================
# PREDICT
# ================================
if st.button("Predict Salary"):

    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)

    # Numeric
    input_df["WorkExp"] = float(exp)
    input_df["YearsCodePro"] = float(code)
    input_df["Age"] = float(age)

    # Categorical encoding
    for col in [
        f"Country_{selected_country}",
        f"EdLevel_{selected_ed}",
        f"RemoteWork_{selected_remote}",
        f"OrgSize_{selected_org}",
        f"DevType_{selected_dev}"
    ]:
        if col in input_df.columns:
            input_df[col] = 1

    # Skills encoding
    for skill in selected_skills:
        if skill in input_df.columns:
            input_df[skill] = 1

    # Clean column names
    input_ready = clean_col_names(input_df.copy())

    # Predict
    prediction = model.predict(input_ready)[0]

    st.success(f"💰 Estimated Salary: ${prediction:,.2f}")
