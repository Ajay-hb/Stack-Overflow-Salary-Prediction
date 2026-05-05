import streamlit as st
import joblib
import pandas as pd

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
# HELPER FUNCTION
# ================================
def extract(prefix):
    return sorted([c.replace(f"{prefix}_", "") for c in MODEL_COLUMNS if c.startswith(f"{prefix}_")])

# Options
countries = extract("Country")
ed_levels = extract("EdLevel")
org_sizes = extract("OrgSize")
remotework = extract("RemoteWork")
employment = extract("Employment")
mainbranch = extract("MainBranch")
aiselect = extract("AISelect")
devtypes = extract("DevType")
languages = extract("LanguageHaveWorkedWith")
databases = extract("DatabaseHaveWorkedWith")
platforms = extract("PlatformHaveWorkedWith")
webframes = extract("WebframeHaveWorkedWith")

# ================================
# UI
# ================================
st.title("💼 Salary Predictor")

exp = st.slider("Work Experience", 0, 50, 5)
code = st.slider("Years Coding", 0, 50, 7)
age = st.slider("Age", 15, 75, 30)

country = st.selectbox("Country", countries)
ed = st.selectbox("Education", ed_levels)
org = st.selectbox("Company Size", org_sizes)
rem = st.selectbox("Remote Work", remotework)
emp = st.selectbox("Employment Type", employment)
branch = st.selectbox("Main Branch", mainbranch)
ais = st.selectbox("AI Adoption", aiselect)

icpm = st.selectbox("IC or Manager", ["Individual contributor", "People manager"])

dev = st.multiselect("Developer Roles", devtypes)
lang = st.multiselect("Languages", languages)
db = st.multiselect("Databases", databases)
plat = st.multiselect("Platforms", platforms)
web = st.multiselect("Web Frameworks", webframes)

# ================================
# PREDICT
# ================================
if st.button("Predict Salary"):

    # Create correct feature vector
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)

    # Numeric
    if "WorkExp" in MODEL_COLUMNS:
        input_df["WorkExp"] = float(exp)

    if "YearsCode" in MODEL_COLUMNS:
        input_df["YearsCode"] = float(code)

    if "YearsCodePro" in MODEL_COLUMNS:
        input_df["YearsCodePro"] = float(code)

    if "Age_Numeric" in MODEL_COLUMNS:
        input_df["Age_Numeric"] = float(age)

    if "Age" in MODEL_COLUMNS:
        input_df["Age"] = float(age)

    # ONE-HOT ONLY (IMPORTANT)
    for feat, val in {
        "Country": country,
        "EdLevel": ed,
        "OrgSize": org,
        "RemoteWork": rem,
        "Employment": emp,
        "MainBranch": branch,
        "AISelect": ais
    }.items():
        col = f"{feat}_{val}"
        if col in MODEL_COLUMNS:
            input_df[col] = 1

    # IC or PM
    if icpm == "People manager":
        if "ICorPM_People manager" in MODEL_COLUMNS:
            input_df["ICorPM_People manager"] = 1

    # Multi-select
    multi_map = {
        "DevType": dev,
        "LanguageHaveWorkedWith": lang,
        "DatabaseHaveWorkedWith": db,
        "PlatformHaveWorkedWith": plat,
        "WebframeHaveWorkedWith": web
    }

    for feat, values in multi_map.items():
        for v in values:
            col = f"{feat}_{v}"
            if col in MODEL_COLUMNS:
                input_df[col] = 1

    # Final alignment
    input_df = input_df.reindex(columns=MODEL_COLUMNS, fill_value=0)
    input_df = input_df.astype(float)

    # Predict (NO numpy)
    pred = model.predict(input_df)[0]

    st.success(f"💰 Estimated Salary: ${pred:,.2f}")
