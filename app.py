
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re

def clean_col_names(df):
    # Simplified regex that avoids complex nesting
    df.columns = [re.sub(r'[:"{}\[\]\,]', '', col) for col in df.columns]
    return df

@st.cache_resource
def load_assets():
    model = joblib.load('salary_stack_model.joblib')
    cols = joblib.load('model_columns.joblib')
    return model, cols

try:
    model, model_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

st.title("Advanced Salary Predictor")
st.markdown("Estimate your annual compensation based on the latest developer survey data.")

st.sidebar.header("Professional Profile")
exp = st.sidebar.slider("Years of Experience", 0, 50, 5)
code = st.sidebar.slider("Years of Coding", 0, 50, 7)
age = st.sidebar.number_input("Age", 15.0, 80.0, 30.0)

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("Country", ['Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Brazil', 'Bulgaria', 'Cambodia', 'Cameroon', 'Canada', 'Chile', 'China', 'Colombia', 'Congo, Republic of the...', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', "Côte d'Ivoire", 'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Ethiopia', 'Finland', 'France', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong (S.A.R.)', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran, Islamic Republic of...', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Libyan Arab Jamahiriya', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Nigeria', 'Nomadic', 'North Korea', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Republic of Korea', 'Republic of Moldova', 'Republic of North Macedonia', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Lucia', 'Saudi Arabia', 'Senegal', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom of Great Britain and Northern Ireland', 'United Republic of Tanzania', 'United States of America', 'Uruguay', 'Uzbekistan', 'Venezuela, Bolivarian Republic of...', 'Viet Nam', 'Yemen', 'Zambia', 'Zimbabwe'], index=147)
    ed = st.selectbox("Education", ['Bachelor’s degree (B.A., B.S., B.Eng., etc.)', 'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)', 'Other (please specify):', 'Primary/elementary school', 'Professional degree (JD, MD, Ph.D, Ed.D, etc.)', 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)', 'Some college/university study without earning a degree'])
    org = st.selectbox("Company Size", ['10,000 or more employees', '100 to 499 employees', '20 to 99 employees', '5,000 to 9,999 employees', '500 to 999 employees', 'I don’t know', 'Just me - I am a freelancer, sole proprietor, etc.', 'Less than 20 employees'])
    rem = st.selectbox("Work Type", ['Hybrid (some remote, leans heavy to in-person)', 'In-person', 'Remote', 'Your choice (very flexible, you can come in when you want or just as needed)'])

with col2:
    emp = st.selectbox("Employment", ['I prefer not to say', 'Independent contractor, freelancer, or self-employed', 'Not employed', 'Retired', 'Student'])
    branch = st.selectbox("Main Branch", ['I am learning to code', 'I am not primarily a developer, but I write code sometimes as part of my work/studies', 'I code primarily as a hobby', 'I used to be a developer by profession, but no longer am', 'I work with developers or my work supports developers but am not a developer by profession'])
    icpm = st.selectbox("Role Level", ['Individual contributor', 'People manager'])
    ais = st.selectbox("AI Adoption", ['No, but I plan to soon', 'Yes, I use AI tools daily', 'Yes, I use AI tools monthly or infrequently', 'Yes, I use AI tools weekly'])

st.divider()
st.subheader("Technical Stack & Roles")

dev = st.multiselect("Developer Roles", ['AI/ML engineer', 'Academic researcher', 'Applied scientist', 'Architect, software or solutions', 'Cloud infrastructure engineer', 'Cybersecurity or InfoSec professional', 'Data engineer', 'Data or business analyst', 'Data scientist', 'Database administrator or engineer', 'DevOps engineer or professional', 'Developer, AI apps or physical AI', 'Developer, QA or test', 'Developer, back-end', 'Developer, desktop or enterprise applications', 'Developer, embedded applications or devices', 'Developer, front-end', 'Developer, full-stack', 'Developer, game or graphics', 'Developer, mobile', 'Engineering manager', 'Financial analyst or engineer', 'Founder, technology or otherwise', 'Other (please specify):', 'Product manager', 'Project manager', 'Retired', 'Senior executive (C-suite, VP, etc.)', 'Student', 'Support engineer or analyst', 'System administrator', 'UX, Research Ops or UI design professional'])
lang = st.multiselect("Programming Languages", ['Ada', 'Assembly', 'Bash/Shell (all shells)', 'C', 'C#', 'C++', 'COBOL', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Fortran', 'GDScript', 'Gleam', 'Go', 'Groovy', 'HTML/CSS', 'Java', 'JavaScript', 'Kotlin', 'Lisp', 'Lua', 'MATLAB', 'MicroPython', 'Mojo', 'OCaml', 'PHP', 'Perl', 'PowerShell', 'Prolog', 'Python', 'R', 'Ruby', 'Rust', 'SQL', 'Scala', 'Swift', 'TypeScript', 'VBA', 'Visual Basic (.Net)', 'Zig'])
db = st.multiselect("Databases", ['Amazon Redshift', 'BigQuery', 'Cassandra', 'Clickhouse', 'Cloud Firestore', 'Cockroachdb', 'Cosmos DB', 'Databricks SQL', 'Datomic', 'DuckDB', 'Dynamodb', 'Elasticsearch', 'Firebase Realtime Database', 'H2', 'IBM DB2', 'InfluxDB', 'MariaDB', 'Microsoft Access', 'Microsoft SQL Server', 'MongoDB', 'MySQL', 'Neo4J', 'Oracle', 'Pocketbase', 'PostgreSQL', 'Redis', 'SQLite', 'Snowflake', 'Supabase', 'Valkey'])
plat = st.multiselect("Platforms", ['APT', 'Amazon Web Services (AWS)', 'Ansible', 'Bun', 'Cargo', 'Chocolatey', 'Cloudflare', 'Composer', 'Datadog', 'Digital Ocean', 'Docker', 'Firebase', 'Google Cloud', 'Gradle', 'Heroku', 'Homebrew', 'IBM Cloud', 'Kubernetes', 'MSBuild', 'Make', 'Maven (build tool)', 'Microsoft Azure', 'Netlify', 'New Relic', 'Ninja', 'NuGet', 'Pacman', 'Pip', 'Podman', 'Poetry', 'Prometheus', 'Railway', 'Splunk', 'Supabase', 'Terraform', 'Vercel', 'Vite', 'Webpack', 'Yandex Cloud', 'Yarn', 'npm', 'pnpm'])
web = st.multiselect("Web Frameworks", ['ASP.NET', 'ASP.NET Core', 'Angular', 'AngularJS', 'Astro', 'Axum', 'Blazor', 'Deno', 'Django', 'Drupal', 'Express', 'FastAPI', 'Fastify', 'Flask', 'Laravel', 'NestJS', 'Next.js', 'Node.js', 'Nuxt.js', 'Phoenix', 'React', 'Ruby on Rails', 'Spring Boot', 'Svelte', 'Symfony', 'Vue.js', 'WordPress', 'jQuery'])

if st.button("Predict My Salary", type="primary"):
    input_row = pd.DataFrame(0, index=[0], columns=model_columns)
    input_row['WorkExp'] = float(exp)
    input_row['YearsCode'] = float(code)
    input_row['Age_Numeric'] = float(age)

    # Single select categorical mapping
    for feat, val in [('Country', country), ('EdLevel', ed), ('OrgSize', org),
                     ('RemoteWork', rem), ('Employment', emp), ('MainBranch', branch), ('AISelect', ais)]:
        c_name = f"{feat}_{val}"
        if c_name in input_row.columns: input_row[c_name] = 1

    if icpm == 'People manager':
        if 'ICorPM_People manager' in input_row.columns: input_row['ICorPM_People manager'] = 1

    # Multi select technical mapping
    for feat, selections in [('DevType', dev), ('LanguageHaveWorkedWith', lang),
                            ('DatabaseHaveWorkedWith', db), ('PlatformHaveWorkedWith', plat),
                            ('WebframeHaveWorkedWith', web)]:
        for s in selections:
            c_name = f"{feat}_{s}"
            if c_name in input_row.columns: input_row[c_name] = 1

    input_ready = clean_col_names(input_row.copy())
    pred = model.predict(input_ready)[0]
    st.success(f"### Predicted Annual Salary: ${pred:,.2f} USD")
    st.balloons()
