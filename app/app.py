"""
Telecom Customer Churn Prediction — Enterprise Dashboard
Source of truth: Telecom_Churn_Project (notebook v10) + telecom_churn_Project.pkl + scaler.pkl

VERIFIED 2026-07-03: this exact model/scaler pair reproduces the notebook's final evaluation
(cell 83) bit-for-bit: accuracy 87.56168037341632%, confusion matrix [[17248,742],[2056,2449]].
Reconstruction method: fillna(median/mode) -> drop Customer_ID -> get_dummies(drop_first=True)
-> reindex to scaler.feature_names_in_ -> train_test_split(test_size=0.30, random_state=42)
-> scaler.transform(x_test) -> model.predict(x_test_scaled). See Model Performance page.

PRESERVED EXACTLY (do not modify without a verified bug):
    - Model / scaler loading + type check
    - FEATURE_ORDER sourced from scaler.feature_names_in_
    - get_dummies(drop_first=True) encoding
    - Reindex-to-FEATURE_ORDER logic (missing-column fill, extra-column warning)
    - StandardScaler.transform on all 70 columns
    - model.predict / model.predict_proba logic
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
PRIMARY_BLUE = "#0B5FFF"
DARK_BLUE = "#0A2E63"
ACCENT_RED = "#E8483D"
ACCENT_GREEN = "#1CAA6E"
GRAY_TEXT = "#5B6472"

st.markdown(f"""
<style>
    .stApp {{ background-color: #F7F9FC; }}
    section[data-testid="stSidebar"] {{ background-color: {DARK_BLUE}; }}
    section[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF; border: 1px solid #E4E9F2; border-radius: 10px;
        padding: 14px 18px; box-shadow: 0 1px 3px rgba(10,46,99,0.06);
    }}
    .kpi-card {{
        background-color: #FFFFFF; border-radius: 12px; padding: 22px;
        border: 1px solid #E4E9F2; box-shadow: 0 1px 4px rgba(10,46,99,0.06); text-align: center;
    }}
    .kpi-value {{ font-size: 30px; font-weight: 700; color: {DARK_BLUE}; }}
    .kpi-label {{ font-size: 13px; color: {GRAY_TEXT}; text-transform: uppercase; letter-spacing: .04em; }}
    .section-header {{ color: {DARK_BLUE}; border-left: 5px solid {PRIMARY_BLUE}; padding-left: 12px; margin: 6px 0 18px 0; }}
    .result-card-stay {{ background: linear-gradient(135deg, #E9FBF3, #FFFFFF); border: 1px solid {ACCENT_GREEN}; border-radius: 14px; padding: 28px; }}
    .result-card-churn {{ background: linear-gradient(135deg, #FDECEA, #FFFFFF); border: 1px solid {ACCENT_RED}; border-radius: 14px; padding: 28px; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load artifacts — UNCHANGED PIPELINE LOGIC
# ----------------------------------------------------------------------------
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "telecom_churn_Project.pkl")
    scaler = joblib.load(MODEL_DIR / "Scaler.pkl")

    expected_type = "RandomForestClassifier"
    actual_type = type(model).__name__

    if actual_type != expected_type:
        st.error(
            f"Loaded model is {actual_type}, not {expected_type}."
        )
        st.stop()

    return model, scaler

model, scaler = load_artifacts()
FEATURE_ORDER = list(scaler.feature_names_in_)  # ground truth, 70 columns


@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "dataset" / "telecom_churn_data.csv")


df_raw = load_data()

# ----------------------------------------------------------------------------
# Category options — UNCHANGED
# ----------------------------------------------------------------------------
GENDER_OPTIONS = ["Female", "Male"]
MARITAL_OPTIONS = ["Divorced", "Married", "Single"]
CITY_OPTIONS = ["Chicago", "Dallas", "Houston", "Los Angeles", "New York",
                "Philadelphia", "Phoenix", "San Antonio", "San Diego", "San Jose"]
STATE_OPTIONS = ["Arizona", "California", "Illinois", "New York", "Pennsylvania", "Texas"]
OCCUPATION_OPTIONS = ["Business Owner", "Government", "Healthcare", "IT/Tech",
                       "Professional", "Retired", "Service Worker", "Student"]
EDUCATION_OPTIONS = ["Bachelor's", "High School", "Master's", "PhD", "Some College"]
PLAN_TYPE_OPTIONS = ["Business", "Family", "Postpaid", "Prepaid"]
CONTRACT_TYPE_OPTIONS = ["Month-to-Month", "One-Year", "Two-Year"]
PAYMENT_METHOD_OPTIONS = ["Bank Transfer", "Check", "Credit Card", "Debit Card", "Digital Wallet"]
APP_USAGE_OPTIONS = ["High", "Low", "Medium"]
REVENUE_TREND_OPTIONS = ["Declining", "Growing", "Stable"]

# ----------------------------------------------------------------------------
# Verified historical model comparison — sourced from notebook cell 77 output,
# cross-checked against individual per-model accuracy_score() print statements.
# Static by design: this is a one-time training experiment result, not something
# to recompute on every app run.
# ----------------------------------------------------------------------------
MODEL_COMPARISON = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression", "SVM", "KNN", "Decision Tree", "Naive Bayes"],
    "Accuracy (%)": [87.5617, 87.4239, 87.2549, 83.8942, 81.7204, 77.5550],
})

# ----------------------------------------------------------------------------
# Live evaluation set reconstruction — deterministic, matches notebook exactly
# (fillna -> get_dummies -> reindex -> split(random_state=42, test_size=0.30)).
# Verified to reproduce accuracy 87.56168037341632% and the notebook's exact
# confusion matrix before this app was shipped.
# ----------------------------------------------------------------------------
@st.cache_data
def evaluate_model():
    d = df_raw.drop(["Customer_ID"], axis=1).copy()
    for col in d.select_dtypes(include=["int64", "float64"]):
        d[col] = d[col].fillna(d[col].median())
    for col in d.select_dtypes(include=["object"]):
        d[col] = d[col].fillna(d[col].mode()[0])

    x = d.iloc[:, :-1]
    y = d.iloc[:, -1]
    x = pd.get_dummies(x, drop_first=True)
    x = x.reindex(columns=FEATURE_ORDER, fill_value=0)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42
    )
    x_test_scaled = scaler.transform(x_test)
    y_pred = model.predict(x_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return acc, cm, report, len(y_train), len(y_test)


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 📡 Churn Prediction")
st.sidebar.caption("ML Classification Project")
st.sidebar.divider()

PAGES = [
    "🏠 Home",
    "📊 Dataset Overview",
    "📈 Analytics Dashboard",
    "🤖 Customer Churn Prediction",
    "📉 Model Performance",
    "⭐ Feature Importance",
    "ℹ️ About Project",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Best Model: Random Forest Classifier")
st.sidebar.caption(f"Model Features: {len(FEATURE_ORDER)}")
st.sidebar.caption(f"Dataset Records: {len(df_raw):,}")


def kpi_card(label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown(f"<h1 style='color:{DARK_BLUE};'>Telecom Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{GRAY_TEXT}; font-size:16px;'>A Machine Learning Classification Project</p>",
                unsafe_allow_html=True)
    st.divider()

    acc, cm, report, n_train, n_test = evaluate_model()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Best Model", "Random Forest")
    with c2: kpi_card("Test Accuracy", f"{acc*100:.2f}%")
    with c3: kpi_card("Records", f"{len(df_raw):,}")
    with c4: kpi_card("Model Features", f"{len(FEATURE_ORDER)}")

    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("<h4 class='section-header'>Project Objective</h4>", unsafe_allow_html=True)
        st.write(
            "The objective of this project is to predict whether a telecom customer is likely "
            "to churn, based on customer demographics, service usage, billing information, and "
            "customer satisfaction. Multiple machine learning classification models were trained "
            "and compared to identify the best-performing algorithm for this task."
        )
        st.markdown("**Dataset**")
        st.write(
            f"Telecom Customer Churn Dataset — {len(df_raw):,} customer records, "
            f"{df_raw.shape[1]-2} raw predictive attributes (excluding customer ID and target), "
            f"expanded to {len(FEATURE_ORDER)} model features after one-hot encoding."
        )
        st.markdown("**Algorithms Compared**")
        st.write("Logistic Regression · Decision Tree · Random Forest · K-Nearest Neighbors · "
                  "Support Vector Machine · Naive Bayes")
        st.markdown("**Technologies Used**")
        st.write("Python · scikit-learn · pandas · NumPy · Streamlit · Plotly · Joblib")
    with c2:
        churn_counts = df_raw["Churn_Flag"].value_counts().sort_index()
        fig = go.Figure(data=[go.Pie(
            labels=["Stayed", "Churned"],
            values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
            hole=0.55,
            marker_colors=[ACCENT_GREEN, ACCENT_RED],
        )])
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260,
                           showlegend=True, title="Target Class Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DATASET OVERVIEW
# ============================================================================
elif page == "📊 Dataset Overview":
    st.markdown(f"<h2 style='color:{DARK_BLUE};'>Dataset Overview</h2>", unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Rows", f"{df_raw.shape[0]:,}")
    with c2: kpi_card("Columns", f"{df_raw.shape[1]}")
    with c3: kpi_card("Missing Values", f"{int(df_raw.isnull().sum().sum())}")
    with c4: kpi_card("Duplicate Rows", f"{int(df_raw.duplicated().sum())}")

    dup_count = int(df_raw.duplicated().sum())
    if dup_count > 0:
        st.caption(
            f"Note: {dup_count} duplicate rows are present in the raw dataset and were not "
            "removed during preprocessing — the notebook detects duplicates but trains on the "
            "full dataset including them."
        )

    st.write("")
    st.markdown("<h4 class='section-header'>Preview</h4>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h4 class='section-header'>Data Types</h4>", unsafe_allow_html=True)
        dtypes_df = df_raw.dtypes.astype(str).reset_index()
        dtypes_df.columns = ["Column", "Type"]
        st.dataframe(dtypes_df, use_container_width=True, height=320)
    with c2:
        st.markdown("<h4 class='section-header'>Target Distribution</h4>", unsafe_allow_html=True)
        churn_counts = df_raw["Churn_Flag"].value_counts().sort_index()
        fig = px.bar(
            x=["Stayed (0)", "Churned (1)"],
            y=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
            color=["Stayed (0)", "Churned (1)"],
            color_discrete_sequence=[ACCENT_GREEN, ACCENT_RED],
            labels={"x": "", "y": "Customers"},
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ANALYTICS DASHBOARD
# ============================================================================
elif page == "📈 Analytics Dashboard":
    st.markdown(f"<h2 style='color:{DARK_BLUE};'>Analytics Dashboard</h2>", unsafe_allow_html=True)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Churn Distribution**")
        churn_counts = df_raw["Churn_Flag"].value_counts().sort_index()
        fig = px.pie(names=["Stayed", "Churned"],
                     values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                     color_discrete_sequence=[ACCENT_GREEN, ACCENT_RED], hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Churn by Plan Type**")
        ct = pd.crosstab(df_raw["Plan_Type"], df_raw["Churn_Flag"], normalize="index") * 100
        ct = ct.rename(columns={0: "Stayed %", 1: "Churned %"}).reset_index()
        fig = px.bar(ct, x="Plan_Type", y=["Stayed %", "Churned %"],
                     color_discrete_sequence=[ACCENT_GREEN, ACCENT_RED], barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Age Distribution**")
        fig = px.histogram(df_raw, x="Age", nbins=30, color_discrete_sequence=[PRIMARY_BLUE])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Gender Distribution**")
        fig = px.bar(df_raw["Gender"].value_counts().reset_index(),
                     x="Gender", y="count", color_discrete_sequence=[PRIMARY_BLUE])
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Complaint Count Distribution**")
        fig = px.histogram(df_raw, x="Complaint_Count", color="Churn_Flag",
                            color_discrete_sequence=[ACCENT_GREEN, ACCENT_RED],
                            barmode="overlay", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Satisfaction Score Distribution**")
        fig = px.histogram(df_raw, x="Satisfaction_Score", color="Churn_Flag",
                            color_discrete_sequence=[ACCENT_GREEN, ACCENT_RED],
                            barmode="overlay", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Correlation Heatmap (Numeric Features)**")
    numeric_df = df_raw.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CHURN PREDICTION — pipeline logic UNCHANGED, layout only
# ============================================================================
elif page == "🤖 Customer Churn Prediction":
    st.markdown(f"<h2 style='color:{DARK_BLUE};'>Customer Churn Prediction</h2>", unsafe_allow_html=True)
    st.caption("Fill in customer details across each tab, then run the prediction.")
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📌 Demographics", "📌 Customer Plan", "📌 Usage Information",
         "📌 Billing Information", "📌 Customer Experience"]
    )

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=90, value=38)
            gender = st.selectbox("Gender", GENDER_OPTIONS)
        with c2:
            marital_status = st.selectbox("Marital Status", MARITAL_OPTIONS)
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)
        with c3:
            city = st.selectbox("City", CITY_OPTIONS)
            state = st.selectbox("State", STATE_OPTIONS)
        with c4:
            occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
            education_level = st.selectbox("Education Level", EDUCATION_OPTIONS)

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=72000.0, step=1000.0)
            tenure_months = st.number_input("Tenure (Months)", min_value=0, max_value=200, value=36)
        with c2:
            plan_type = st.selectbox("Plan Type", PLAN_TYPE_OPTIONS)
            contract_type = st.selectbox("Contract Type", CONTRACT_TYPE_OPTIONS)
        with c3:
            payment_method = st.selectbox("Payment Method", PAYMENT_METHOD_OPTIONS)
            auto_pay = st.selectbox("Auto Pay Enabled", ["No", "Yes"])
        with c4:
            international_plan = st.selectbox("International Plan", ["No", "Yes"])
            family_plan = st.selectbox("Family Plan", ["No", "Yes"])

    with tab3:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            number_of_lines = st.slider("Number of Lines", 1, 15, 2)
            monthly_call_minutes = st.slider("Monthly Call Minutes", 0, 3000, 650)
        with c2:
            monthly_sms = st.slider("Monthly SMS", 0, 900, 300)
            monthly_data_usage_gb = st.slider("Monthly Data Usage (GB)", 0.0, 200.0, 27.0)
        with c3:
            streaming_usage_hours = st.slider("Streaming Usage (Hours)", 0.0, 320.0, 34.0)
            roaming_usage_gb = st.slider("Roaming Usage (GB)", 0.0, 20.0, 1.0)
        with c4:
            app_usage_frequency = st.selectbox("App Usage Frequency", APP_USAGE_OPTIONS)
            last_login_days_ago = st.slider("Last Login (Days Ago)", 0, 180, 20)

    with tab4:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=91.0, step=1.0)
            total_bill_amount = st.number_input("Total Bill Amount ($)", min_value=0.0, value=3480.0, step=10.0)
        with c2:
            arpu = st.number_input("ARPU ($)", min_value=0.0, value=97.0, step=1.0)
            revenue_last_3_months = st.number_input("Revenue Last 3 Months ($)", min_value=0.0, value=288.0, step=10.0)
        with c3:
            customer_lifetime_value = st.number_input("Customer Lifetime Value ($)", min_value=0.0, value=4000.0, step=50.0)
            discount_received = st.number_input("Discount Received ($)", min_value=0.0, value=10.0, step=1.0)
        with c4:
            revenue_trend = st.selectbox("Revenue Trend", REVENUE_TREND_OPTIONS)
            reward_points = st.number_input("Reward Points", min_value=0, value=15000, step=100)

    with tab5:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            customer_service_calls = st.slider("Customer Service Calls", 0, 18, 2)
            complaint_count = st.slider("Complaint Count", 0, 16, 1)
        with c2:
            network_outages_experienced = st.slider("Network Outages Experienced", 0, 13, 2)
            dropped_call_rate = st.slider("Dropped Call Rate (0-1)", 0.0, 1.0, 0.09)
        with c3:
            late_payment_count = st.slider("Late Payment Count", 0, 7, 0)
            missed_payment_count = st.slider("Missed Payment Count", 0, 6, 0)
        with c4:
            satisfaction_score = st.slider("Satisfaction Score (1-5)", 1.0, 5.0, 4.0)
            nps_score = st.slider("NPS Score (-100 to 100)", -100, 100, 38)

    st.write("")
    predict_clicked = st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary")

    # ------------------------------------------------------------------------
    # PREPROCESSING + PREDICTION — UNCHANGED
    # ------------------------------------------------------------------------
    if predict_clicked:
        raw = {
            "Age": age, "Gender": gender, "Marital_Status": marital_status, "Dependents": dependents,
            "City": city, "State": state, "Occupation": occupation, "Education_Level": education_level,
            "Annual_Income": annual_income, "Tenure_Months": tenure_months, "Plan_Type": plan_type,
            "Contract_Type": contract_type, "Monthly_Charges": monthly_charges,
            "Total_Bill_Amount": total_bill_amount, "Payment_Method": payment_method,
            "Auto_Pay": 1 if auto_pay == "Yes" else 0,
            "International_Plan": 1 if international_plan == "Yes" else 0,
            "Roaming_Usage_GB": roaming_usage_gb,
            "Family_Plan": 1 if family_plan == "Yes" else 0,
            "Number_of_Lines": number_of_lines, "Monthly_Call_Minutes": monthly_call_minutes,
            "Monthly_SMS": monthly_sms, "Monthly_Data_Usage_GB": monthly_data_usage_gb,
            "Streaming_Usage_Hours": streaming_usage_hours,
            "Customer_Service_Calls": customer_service_calls, "Complaint_Count": complaint_count,
            "Network_Outages_Experienced": network_outages_experienced,
            "Dropped_Call_Rate": dropped_call_rate, "Last_Login_Days_Ago": last_login_days_ago,
            "App_Usage_Frequency": app_usage_frequency, "Reward_Points": reward_points,
            "Late_Payment_Count": late_payment_count, "Missed_Payment_Count": missed_payment_count,
            "Satisfaction_Score": satisfaction_score, "NPS_Score": nps_score, "ARPU": arpu,
            "Customer_Lifetime_Value": customer_lifetime_value,
            "Revenue_Last_3_Months": revenue_last_3_months, "Revenue_Trend": revenue_trend,
            "Discount_Received": discount_received,
        }

        input_df = pd.DataFrame([raw])
        encoded = pd.get_dummies(input_df, drop_first=True)

        missing_cols = [c for c in FEATURE_ORDER if c not in encoded.columns]
        for c in missing_cols:
            encoded[c] = 0
        encoded = encoded[FEATURE_ORDER]

        extra_cols = [c for c in encoded.columns if c not in FEATURE_ORDER]
        if extra_cols:
            st.warning(f"Unexpected columns generated and dropped: {extra_cols}. "
                        f"Check that selectbox options match training data categories exactly.")

        scaled = scaler.transform(encoded)
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0]
        churn_prob = proba[1]
        stay_prob = proba[0]

        st.divider()
        st.markdown("<h3 class='section-header'>Prediction Result</h3>", unsafe_allow_html=True)

        if pred == 1:
            risk_level = "HIGH" if churn_prob >= 0.7 else "MODERATE"
            st.markdown(f"""
            <div class="result-card-churn">
                <h3 style="color:{ACCENT_RED}; margin-top:0;">⚠️ Customer Likely to Churn</h3>
                <p style="font-size:15px; color:{GRAY_TEXT};">Confidence Score: <b>{churn_prob*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: kpi_card("Churn Probability", f"{churn_prob*100:.1f}%")
            with c2: kpi_card("Stay Probability", f"{stay_prob*100:.1f}%")
            with c3: kpi_card("Risk Level", risk_level)
            st.write("")
            st.markdown("**⚠️ High Risk Customer — Recommended Actions**")
            st.markdown("- Offer a targeted discount or loyalty rate\n"
                        "- Proactive contact from a retention specialist\n"
                        "- Escalate to priority support queue\n"
                        "- Award bonus reward points\n"
                        "- Enroll in a structured retention campaign")
        else:
            st.markdown(f"""
            <div class="result-card-stay">
                <h3 style="color:{ACCENT_GREEN}; margin-top:0;">✅ Customer Likely to Stay</h3>
                <p style="font-size:15px; color:{GRAY_TEXT};">Confidence Score: <b>{stay_prob*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: kpi_card("Stay Probability", f"{stay_prob*100:.1f}%")
            with c2: kpi_card("Churn Probability", f"{churn_prob*100:.1f}%")
            with c3: kpi_card("Risk Level", "LOW")
            st.write("")
            st.markdown("**✅ Customer is Stable — Suggested Actions**")
            st.markdown("- Continue standard loyalty rewards\n"
                        "- Explore premium plan upsell opportunities\n"
                        "- Maintain regular engagement touchpoints")

        st.progress(min(max(float(churn_prob), 0.0), 1.0))

        with st.expander("Show model input (post-encoding, pre-scaling)"):
            st.dataframe(encoded.T.rename(columns={0: "value"}), use_container_width=True)

# ============================================================================
# MODEL PERFORMANCE
# ============================================================================
elif page == "📉 Model Performance":
    st.markdown(f"<h2 style='color:{DARK_BLUE};'>Model Performance</h2>", unsafe_allow_html=True)
    st.caption(
        "Computed live from the deployed model and scaler against a deterministic 70/30 split "
        "(random_state=42), reproducing the notebook's final evaluation exactly."
    )
    st.divider()

    acc, cm, report, n_train, n_test = evaluate_model()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Accuracy", f"{acc*100:.2f}%")
    with c2: kpi_card("Train Set Size", f"{n_train:,}")
    with c3: kpi_card("Test Set Size", f"{n_test:,}")
    with c4: kpi_card("Macro F1", f"{report['macro avg']['f1-score']*100:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix**")
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         x=["Predicted: Stay", "Predicted: Churn"],
                         y=["Actual: Stay", "Actual: Churn"])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True, height=400)

    st.markdown("<h4 class='section-header'>Model Comparison</h4>", unsafe_allow_html=True)
    st.caption("Accuracy for all six algorithms evaluated in the notebook, on the same train/test split.")
    fig = px.bar(MODEL_COMPARISON.sort_values("Accuracy (%)"), x="Accuracy (%)", y="Model",
                 orientation="h", color="Accuracy (%)", color_continuous_scale="Blues")
    fig.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(MODEL_COMPARISON.sort_values("Accuracy (%)", ascending=False),
                 use_container_width=True, hide_index=True)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
elif page == "⭐ Feature Importance":
    st.markdown(f"<h2 style='color:{DARK_BLUE};'>Feature Importance</h2>", unsafe_allow_html=True)
    st.divider()

    importance_df = pd.DataFrame({
        "Feature": FEATURE_ORDER,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(10)

    fig = px.bar(importance_df.sort_values("Importance"), x="Importance", y="Feature",
                 orientation="h", color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h4 class='section-header'>What These Features Mean</h4>", unsafe_allow_html=True)
    explanations = {
        "Complaint_Count": "How many complaints a customer has filed — repeated complaints signal unresolved friction.",
        "Satisfaction_Score": "Self-reported satisfaction rating — the most direct proxy for churn intent available.",
        "NPS_Score": "Net Promoter Score — captures whether a customer would recommend the service, a leading indicator of loyalty.",
        "Customer_Service_Calls": "Frequency of support contact — high call volume often means unresolved issues.",
        "Dropped_Call_Rate": "Technical service quality — a directly experienced pain point that drives frustration.",
        "Reward_Points": "Accumulated loyalty points — can reflect tenure and engagement depth.",
        "Discount_Received": "Prior discounting — may indicate past retention interventions or price sensitivity.",
        "Customer_Lifetime_Value": "Estimated total value of the customer relationship — higher-value customers may behave differently.",
        "Total_Bill_Amount": "Cumulative billing — reflects both usage and pricing tier.",
        "Network_Outages_Experienced": "Reliability issues experienced directly — erodes trust in the network.",
    }
    for _, row in importance_df.sort_values("Importance", ascending=False).iterrows():
        feat = row["Feature"]
        note = explanations.get(feat, "Contributes to the model's churn prediction based on historical patterns.")
        st.markdown(f"**{feat}** ({row['Importance']*100:.1f}%) — {note}")

# ============================================================================
# ABOUT
# ============================================================================
elif page == "ℹ️ About Project":
    st.markdown(f"<h2 style='color:{DARK_BLUE};'>About This Project</h2>", unsafe_allow_html=True)
    st.divider()

    st.markdown(f"""
    **Project Name:** Telecom Customer Churn Prediction

    **Project Type:** Machine Learning Classification Project (academic)

    **Business Problem:** Customer churn directly erodes recurring revenue in subscription-based
    telecom businesses. Identifying customers at risk of churning before they leave allows a
    business to intervene with targeted retention actions rather than reacting after the fact.

    **Objective:** Predict whether a telecom customer is likely to churn, using demographic,
    billing, usage, and service-experience data, and identify which factors drive that risk.

    **Dataset:** {len(df_raw):,} customer records, {df_raw.shape[1]} raw columns (including
    customer ID and target label). {int(df_raw.duplicated().sum())} duplicate rows were identified
    during exploratory analysis; the notebook trains on the full dataset without removing them.

    **Machine Learning Workflow:**
    1. Data loading and shape/type inspection
    2. Exploratory data analysis — distributions, churn breakdowns, correlation analysis
    3. Data cleaning — median imputation for numeric fields, mode imputation for categorical fields
    4. Drop `Customer_ID` (non-predictive identifier)
    5. Feature/target split
    6. One-hot encoding with `drop_first=True` to avoid the dummy variable trap
    7. Train/test split — 70% train, 30% test, `random_state=42`
    8. Feature scaling — `StandardScaler` fit on the training split, applied to all features
    9. Model training and comparison across six algorithms
    10. Model selection — Random Forest chosen on test-set accuracy
    11. Model evaluation — confusion matrix, classification report
    12. Feature importance analysis
    13. Model and scaler persistence with Joblib for deployment

    **Algorithms Compared:** Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors,
    Support Vector Machine, Naive Bayes

    **Best Model:** Random Forest Classifier — {87.5617:.2f}% test accuracy

    **Libraries Used:** pandas, NumPy, scikit-learn, Streamlit, Plotly, Joblib
    (notebook also used matplotlib and seaborn for exploratory visualization)

    **Future Improvements:**
    - Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) on the Random Forest
    - Resolve the detected-but-unremoved duplicate records — decide explicitly whether to drop them
    - SHAP-based per-customer explanation instead of only global feature importance
    - Class imbalance handling (SMOTE or class weighting) — recall on the churn class currently
      lags precision, meaning some churners are being missed
    - Model monitoring for data or concept drift in production
    """)
