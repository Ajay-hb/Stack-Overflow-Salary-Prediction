import re

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Advanced Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


def clean_col_names(df):
    df.columns = [re.sub(r'[:"{}\[\]\,]', "", col) for col in df.columns]
    return df


@st.cache_resource
def load_assets():
    model = joblib.load("salary_stack_model.joblib")
    cols = joblib.load("model_columns.joblib")
    return model, cols


try:
    model, model_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()


COUNTRIES = [
    "Albania", "Algeria", "Andorra", "Angola", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Brazil", "Bulgaria", "Cambodia", "Cameroon",
    "Canada", "Chile", "China", "Colombia", "Congo, Republic of the...", "Costa Rica", "Croatia",
    "Cuba", "Cyprus", "Czech Republic", "Côte d'Ivoire", "Democratic Republic of the Congo",
    "Denmark", "Djibouti", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Estonia",
    "Ethiopia", "Finland", "France", "Georgia", "Germany", "Ghana", "Greece", "Guatemala",
    "Guyana", "Haiti", "Honduras", "Hong Kong (S.A.R.)", "Hungary", "Iceland", "India",
    "Indonesia", "Iran, Islamic Republic of...", "Iraq", "Ireland", "Isle of Man", "Israel",
    "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kosovo", "Kuwait",
    "Kyrgyzstan", "Latvia", "Lebanon", "Libyan Arab Jamahiriya", "Lithuania", "Luxembourg",
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Mauritania", "Mauritius",
    "Mexico", "Moldova", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Nigeria", "Nomadic",
    "North Korea", "Norway", "Oman", "Pakistan", "Panama", "Papua New Guinea", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Republic of Korea",
    "Republic of Moldova", "Republic of North Macedonia", "Romania", "Russian Federation",
    "Rwanda", "Saint Lucia", "Saudi Arabia", "Senegal", "Serbia", "Singapore", "Slovakia",
    "Slovenia", "Somalia", "South Africa", "South Korea", "Spain", "Sri Lanka", "Sudan",
    "Suriname", "Swaziland", "Sweden", "Switzerland", "Syrian Arab Republic", "Taiwan",
    "Thailand", "Togo", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Uganda",
    "Ukraine", "United Arab Emirates", "United Kingdom of Great Britain and Northern Ireland",
    "United Republic of Tanzania", "United States of America", "Uruguay", "Uzbekistan",
    "Venezuela, Bolivarian Republic of...", "Viet Nam", "Yemen", "Zambia", "Zimbabwe",
]

EDUCATION = [
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
    "Other (please specify):",
    "Primary/elementary school",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
    "Some college/university study without earning a degree",
]

ORG_SIZE = [
    "10,000 or more employees", "100 to 499 employees", "20 to 99 employees",
    "5,000 to 9,999 employees", "500 to 999 employees", "I don’t know",
    "Just me - I am a freelancer, sole proprietor, etc.", "Less than 20 employees",
]

WORK_TYPE = [
    "Hybrid (some remote, leans heavy to in-person)", "In-person", "Remote",
    "Your choice (very flexible, you can come in when you want or just as needed)",
]

EMPLOYMENT = [
    "I prefer not to say", "Independent contractor, freelancer, or self-employed",
    "Not employed", "Retired", "Student",
]

MAIN_BRANCH = [
    "I am learning to code",
    "I am not primarily a developer, but I write code sometimes as part of my work/studies",
    "I code primarily as a hobby",
    "I used to be a developer by profession, but no longer am",
    "I work with developers or my work supports developers but am not a developer by profession",
]

AI_ADOPTION = [
    "No, but I plan to soon", "Yes, I use AI tools daily",
    "Yes, I use AI tools monthly or infrequently", "Yes, I use AI tools weekly",
]

DEV_ROLES = [
    "AI/ML engineer", "Academic researcher", "Applied scientist", "Architect, software or solutions",
    "Cloud infrastructure engineer", "Cybersecurity or InfoSec professional", "Data engineer",
    "Data or business analyst", "Data scientist", "Database administrator or engineer",
    "DevOps engineer or professional", "Developer, AI apps or physical AI", "Developer, QA or test",
    "Developer, back-end", "Developer, desktop or enterprise applications",
    "Developer, embedded applications or devices", "Developer, front-end", "Developer, full-stack",
    "Developer, game or graphics", "Developer, mobile", "Engineering manager",
    "Financial analyst or engineer", "Founder, technology or otherwise", "Other (please specify):",
    "Product manager", "Project manager", "Retired", "Senior executive (C-suite, VP, etc.)",
    "Student", "Support engineer or analyst", "System administrator", "UX, Research Ops or UI design professional",
]

LANGUAGES = [
    "Ada", "Assembly", "Bash/Shell (all shells)", "C", "C#", "C++", "COBOL", "Dart", "Delphi",
    "Elixir", "Erlang", "F#", "Fortran", "GDScript", "Gleam", "Go", "Groovy", "HTML/CSS",
    "Java", "JavaScript", "Kotlin", "Lisp", "Lua", "MATLAB", "MicroPython", "Mojo", "OCaml",
    "PHP", "Perl", "PowerShell", "Prolog", "Python", "R", "Ruby", "Rust", "SQL", "Scala",
    "Swift", "TypeScript", "VBA", "Visual Basic (.Net)", "Zig",
]

DATABASES = [
    "Amazon Redshift", "BigQuery", "Cassandra", "Clickhouse", "Cloud Firestore", "Cockroachdb",
    "Cosmos DB", "Databricks SQL", "Datomic", "DuckDB", "Dynamodb", "Elasticsearch",
    "Firebase Realtime Database", "H2", "IBM DB2", "InfluxDB", "MariaDB", "Microsoft Access",
    "Microsoft SQL Server", "MongoDB", "MySQL", "Neo4J", "Oracle", "Pocketbase", "PostgreSQL",
    "Redis", "SQLite", "Snowflake", "Supabase", "Valkey",
]

PLATFORMS = [
    "APT", "Amazon Web Services (AWS)", "Ansible", "Bun", "Cargo", "Chocolatey", "Cloudflare",
    "Composer", "Datadog", "Digital Ocean", "Docker", "Firebase", "Google Cloud", "Gradle",
    "Heroku", "Homebrew", "IBM Cloud", "Kubernetes", "MSBuild", "Make", "Maven (build tool)",
    "Microsoft Azure", "Netlify", "New Relic", "Ninja", "NuGet", "Pacman", "Pip", "Podman",
    "Poetry", "Prometheus", "Railway", "Splunk", "Supabase", "Terraform", "Vercel", "Vite",
    "Webpack", "Yandex Cloud", "Yarn", "npm", "pnpm",
]

WEB_FRAMEWORKS = [
    "ASP.NET", "ASP.NET Core", "Angular", "AngularJS", "Astro", "Axum", "Blazor", "Deno",
    "Django", "Drupal", "Express", "FastAPI", "Fastify", "Flask", "Laravel", "NestJS",
    "Next.js", "Node.js", "Nuxt.js", "Phoenix", "React", "Ruby on Rails", "Spring Boot",
    "Svelte", "Symfony", "Vue.js", "WordPress", "jQuery",
]


st.markdown(
    """
<style>
:root {
    --page: #080c12;
    --panel: rgba(255,255,255,0.075);
    --panel-2: rgba(255,255,255,0.11);
    --line: rgba(255,255,255,0.15);
    --text: #f6f8fb;
    --muted: #a8b3c3;
    --cyan: #42d9ff;
    --green: #4ff0a5;
    --yellow: #ffd166;
    --coral: #ff6b6b;
}

html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(66,217,255,0.20), transparent 28%),
        radial-gradient(circle at 88% 4%, rgba(79,240,165,0.16), transparent 24%),
        linear-gradient(135deg, #080c12 0%, #101827 48%, #111827 100%);
    color: var(--text);
}

.block-container {
    max-width: 1220px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: rgba(8,12,18,0.92);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

.hero {
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 34px;
    overflow: hidden;
    position: relative;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.13), rgba(255,255,255,0.045)),
        linear-gradient(120deg, rgba(66,217,255,0.16), rgba(79,240,165,0.08));
    box-shadow: 0 28px 90px rgba(0,0,0,0.34);
    animation: fadeUp 720ms ease both;
}

.hero:after {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    right: -90px;
    top: -100px;
    background: conic-gradient(from 130deg, rgba(66,217,255,0.55), rgba(79,240,165,0.36), rgba(255,209,102,0.32), rgba(66,217,255,0.55));
    filter: blur(42px);
    opacity: .42;
    animation: drift 7s ease-in-out infinite;
}

.hero-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(270px, .75fr);
    gap: 28px;
    align-items: center;
}

.eyebrow {
    color: var(--cyan);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    margin: 0;
    font-size: clamp(34px, 5vw, 60px);
    line-height: 1.03;
    letter-spacing: 0;
}

.hero p {
    max-width: 680px;
    color: #cad5e3;
    font-size: 17px;
    line-height: 1.7;
    margin: 18px 0 0;
}

.signal-card {
    background: rgba(8,12,18,0.48);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 18px;
    backdrop-filter: blur(18px);
}

.signal-card strong {
    display: block;
    font-size: 36px;
    line-height: 1;
    color: var(--green);
}

.signal-card span {
    color: var(--muted);
    font-size: 13px;
}

.glass {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 22px;
    box-shadow: 0 22px 70px rgba(0,0,0,0.24);
    animation: fadeUp 760ms ease both;
}

.section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin: 24px 0 12px;
}

.section-title h2 {
    margin: 0;
    font-size: 24px;
}

.pill {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 7px 12px;
    color: #d7e2f0;
    background: rgba(255,255,255,0.06);
    font-size: 12px;
    font-weight: 800;
}

.mini-card {
    min-height: 112px;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 17px;
    background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.045));
    transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
}

.mini-card:hover {
    transform: translateY(-4px);
    border-color: rgba(66,217,255,0.44);
    background: linear-gradient(145deg, rgba(66,217,255,0.14), rgba(255,255,255,0.055));
}

.mini-card span {
    display: block;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 800;
}

.mini-card strong {
    display: block;
    margin-top: 10px;
    font-size: 26px;
    line-height: 1.1;
}

.result {
    border: 1px solid rgba(79,240,165,0.34);
    border-radius: 10px;
    padding: 24px;
    background:
        linear-gradient(135deg, rgba(79,240,165,0.15), rgba(66,217,255,0.10)),
        rgba(255,255,255,0.06);
    box-shadow: 0 24px 80px rgba(79,240,165,0.12);
    animation: popIn 520ms ease both;
}

.salary {
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1;
    color: var(--green);
    font-weight: 900;
    margin: 8px 0 12px;
}

.tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.tag {
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 7px 11px;
    background: rgba(255,255,255,0.07);
    color: #d9e4f2;
    font-size: 12px;
    font-weight: 700;
}

.stButton > button {
    width: 100%;
    min-height: 50px;
    border-radius: 10px;
    border: 1px solid rgba(66,217,255,0.42);
    background: linear-gradient(90deg, var(--cyan), var(--green));
    color: #071017;
    font-weight: 900;
    box-shadow: 0 16px 38px rgba(66,217,255,0.18);
    transition: transform 160ms ease, box-shadow 160ms ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 22px 48px rgba(66,217,255,0.26);
}

label, .stSlider label, .stNumberInput label, .stSelectbox label, .stMultiSelect label {
    color: #dbe6f4 !important;
    font-weight: 750 !important;
}

div[data-baseweb="input"], div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border-color: var(--line) !important;
    border-radius: 10px !important;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 18px;
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes drift {
    0%, 100% { transform: translate3d(0,0,0) rotate(0deg); }
    50% { transform: translate3d(-28px,26px,0) rotate(18deg); }
}

@keyframes popIn {
    from { opacity: 0; transform: scale(.98) translateY(10px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

@media (max-width: 860px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
    .hero {
        padding: 24px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="hero">
    <div class="hero-grid">
        <div>
            <div class="eyebrow">Developer compensation intelligence</div>
            <h1>Advanced Salary Predictor</h1>
            <p>Build a complete professional profile, map your stack, and estimate annual compensation from survey-trained model signals.</p>
        </div>
        <div class="signal-card">
            <strong>USD</strong>
            <span>Annual salary estimate with role, location, experience, and technology inputs.</span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Profile Studio")
    st.caption("Set your core experience details before completing the full profile.")
    st.divider()
    exp = st.slider("Years of Experience", 0, 50, 5)
    code = st.slider("Years of Coding", 0, 50, 7)
    age = st.number_input("Age", 15.0, 80.0, 30.0)
    st.divider()
    st.caption("Tip: adding stack and role details usually improves the model context.")


st.markdown(
    """
<div class="section-title">
    <h2>Professional Snapshot</h2>
    <span class="pill">Core inputs</span>
</div>
""",
    unsafe_allow_html=True,
)

snap_cols = st.columns(4)
snapshot = [
    ("Experience", f"{exp} yrs"),
    ("Coding", f"{code} yrs"),
    ("Age", f"{age:.0f}"),
    ("Model Fields", f"{len(model_columns):,}"),
]

for col, (label, value) in zip(snap_cols, snapshot):
    with col:
        st.markdown(
            f"""
<div class="mini-card">
    <span>{label}</span>
    <strong>{value}</strong>
</div>
""",
            unsafe_allow_html=True,
        )


st.markdown(
    """
<div class="section-title">
    <h2>Career Profile</h2>
    <span class="pill">Work and education</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="glass">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    country_index = COUNTRIES.index("United States of America")
    country = st.selectbox("Country", COUNTRIES, index=country_index)
    ed = st.selectbox("Education", EDUCATION)
    org = st.selectbox("Company Size", ORG_SIZE)
    rem = st.selectbox("Work Type", WORK_TYPE)

with col2:
    emp = st.selectbox("Employment", EMPLOYMENT)
    branch = st.selectbox("Main Branch", MAIN_BRANCH)
    icpm = st.selectbox("Role Level", ["Individual contributor", "People manager"])
    ais = st.selectbox("AI Adoption", AI_ADOPTION)

st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
<div class="section-title">
    <h2>Technical Stack & Roles</h2>
    <span class="pill">Model signals</span>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="glass">', unsafe_allow_html=True)
stack_col1, stack_col2 = st.columns(2)

with stack_col1:
    dev = st.multiselect("Developer Roles", DEV_ROLES)
    lang = st.multiselect("Programming Languages", LANGUAGES, default=["Python", "JavaScript"])
    db = st.multiselect("Databases", DATABASES)

with stack_col2:
    plat = st.multiselect("Platforms", PLATFORMS, default=["Docker"])
    web = st.multiselect("Web Frameworks", WEB_FRAMEWORKS, default=["React"])

st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
<div class="section-title">
    <h2>Prediction</h2>
    <span class="pill">Ready when you are</span>
</div>
""",
    unsafe_allow_html=True,
)

predict = st.button("Predict My Salary", type="primary")

if predict:
    input_row = pd.DataFrame(0, index=[0], columns=model_columns)
    input_row["WorkExp"] = float(exp)
    input_row["YearsCode"] = float(code)
    input_row["Age_Numeric"] = float(age)

    single_selects = [
        ("Country", country),
        ("EdLevel", ed),
        ("OrgSize", org),
        ("RemoteWork", rem),
        ("Employment", emp),
        ("MainBranch", branch),
        ("AISelect", ais),
    ]

    for feat, val in single_selects:
        c_name = f"{feat}_{val}"
        if c_name in input_row.columns:
            input_row[c_name] = 1

    if icpm == "People manager" and "ICorPM_People manager" in input_row.columns:
        input_row["ICorPM_People manager"] = 1

    multi_selects = [
        ("DevType", dev),
        ("LanguageHaveWorkedWith", lang),
        ("DatabaseHaveWorkedWith", db),
        ("PlatformHaveWorkedWith", plat),
        ("WebframeHaveWorkedWith", web),
    ]

    for feat, selections in multi_selects:
        for selection in selections:
            c_name = f"{feat}_{selection}"
            if c_name in input_row.columns:
                input_row[c_name] = 1

    input_ready = clean_col_names(input_row.copy())
    pred = model.predict(input_ready)[0]

    result_col, insight_col = st.columns([1.25, 0.75])

    with result_col:
        st.markdown(
            f"""
<div class="result">
    <div class="eyebrow">Predicted Annual Salary</div>
    <div class="salary">${pred:,.2f}</div>
    <p style="color:#cad5e3;margin:0;">Estimated compensation in USD based on your current profile and technology stack.</p>
    <div class="tag-row">
        <span class="tag">{country}</span>
        <span class="tag">{icpm}</span>
        <span class="tag">{rem.split(" ")[0]}</span>
        <span class="tag">{len(lang)} languages</span>
        <span class="tag">{len(dev)} roles</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with insight_col:
        st.metric("Experience", f"{exp} years")
        st.metric("Coding History", f"{code} years")
        st.metric("Stack Signals", len(dev) + len(lang) + len(db) + len(plat) + len(web))

    st.balloons()
