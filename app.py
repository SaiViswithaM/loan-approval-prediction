import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="🏦 Smart Loan Advisor",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# LOAD FILES
# ==========================================
model = joblib.load("loan_model.pkl")
df = pd.read_csv("loan_train.csv")

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.stButton>button {
    width:100%;
    background:linear-gradient(90deg,#2563eb,#1d4ed8);
    color:white;
    border:none;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover {
    background:#1e40af;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🏦 Smart Loan Advisor")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏦 Loan Assessment",
        "📊 Analytics Dashboard",
        "💰 EMI Calculator",
        "📚 Financial Literacy",
        "⚖ Borrower Rights"
    ]
)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div style="
background:linear-gradient(90deg,#0f172a,#1e3a8a);
padding:25px;
border-radius:20px;
text-align:center;
color:white;
">

<h1>🏦 Smart Loan Advisor</h1>

<h4>
AI Powered Financial Eligibility & Risk Assessment Platform
</h4>

</div>
""", unsafe_allow_html=True)

st.write("")

# ==========================================
# DASHBOARD CARDS
# ==========================================
c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#2563eb,#1d4ed8);
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;">
    <h3>🛡 AI Risk Assessment</h3>
    <p>Smart Risk Evaluation</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#10b981,#059669);
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;">
    <h3>📈 Credit Analysis</h3>
    <p>Financial Insights</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#f59e0b,#ea580c);
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;">
    <h3>💰 Loan Advisor</h3>
    <p>Smart Recommendation</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==========================================
# LOAN ASSESSMENT
# ==========================================
if page == "🏦 Loan Assessment":

    st.subheader("Applicant Information")

    col1,col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female","Male"]
        )

        married = st.selectbox(
            "Married",
            ["No","Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            [0,1,2,3]
        )

        education = st.selectbox(
            "Education",
            ["Graduate","Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["No","Yes"]
        )

    with col2:

        income = st.number_input(
            "Applicant Income",
            min_value=0
        )

        co_income = st.number_input(
            "Coapplicant Income",
            min_value=0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0
        )

        loan_term = st.number_input(
            "Loan Term",
            value=360
        )

        credit_history = st.selectbox(
            "Credit History",
            [0,1]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural","Semiurban","Urban"]
        )

    if st.button("🔍 Analyze Applicant"):

        gender_val = 1 if gender=="Male" else 0
        married_val = 1 if married=="Yes" else 0
        education_val = 0 if education=="Graduate" else 1
        self_emp_val = 1 if self_employed=="Yes" else 0

        area_map = {
            "Rural":0,
            "Semiurban":1,
            "Urban":2
        }

        area_val = area_map[property_area]

        data = np.array([[
            gender_val,
            married_val,
            dependents,
            education_val,
            self_emp_val,
            income,
            co_income,
            loan_amount,
            loan_term,
            credit_history,
            area_val
        ]])

        prediction = model.predict(data)
        prob = model.predict_proba(data)[0]

        if loan_amount >= 250:
            loan_type = "🏠 Home Loan"
        elif loan_amount >= 100:
            loan_type = "🚗 Vehicle Loan"
        else:
            loan_type = "💼 Personal Loan"

        if prediction[0] == 1:

            st.balloons()

            st.success("✅ Applicant Appears Eligible")

            risk = "🟢 Low Risk"

        else:

            st.error("❌ Applicant Requires Review")

            risk = "🟡 Medium Risk"

        st.markdown("## 📋 Applicant Assessment Report")

        r1,r2,r3 = st.columns(3)

        with r1:
            st.metric("Risk Level", risk)

        with r2:
            st.metric(
                "Confidence",
                f"{prob[1]*100:.2f}%"
            )

        with r3:
            st.metric(
                "Recommended Loan",
                loan_type
            )

        st.progress(int(prob[1]*100))

        st.info(
            f"Approval Confidence : {prob[1]*100:.2f}%"
        )

# ==========================================
# ANALYTICS
# ==========================================
elif page == "📊 Analytics Dashboard":

    st.subheader("Dataset Analytics")

    fig1 = px.histogram(
        df,
        x="ApplicantIncome",
        title="Applicant Income Distribution"
    )

    st.plotly_chart(
        fig1,
        width="stretch"
    )

    fig2 = px.histogram(
        df,
        x="LoanAmount",
        title="Loan Amount Distribution"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )

    fig3 = px.pie(
        df,
        names="Loan_Status",
        title="Loan Approval Distribution"
    )

    st.plotly_chart(
        fig3,
        width="stretch"
    )

# ==========================================
# EMI CALCULATOR
# ==========================================
elif page == "💰 EMI Calculator":

    st.subheader("💰 EMI Calculator")

    principal = st.number_input(
        "Loan Amount",
        min_value=1000
    )

    rate = st.number_input(
        "Interest Rate (%)",
        value=8.5
    )

    years = st.number_input(
        "Tenure (Years)",
        min_value=1
    )

    if st.button("Calculate EMI"):

        monthly_rate = rate/(12*100)

        months = years*12

        emi = (
            principal *
            monthly_rate *
            (1+monthly_rate)**months
        ) / (
            (1+monthly_rate)**months - 1
        )

        st.success(
            f"Monthly EMI : ₹ {emi:.2f}"
        )

# ==========================================
# FINANCIAL LITERACY
# ==========================================
elif page == "📚 Financial Literacy":

    st.subheader("📚 Financial Literacy")

    st.info("""
✔ Maintain a good credit score

✔ Borrow only what you can repay

✔ Compare interest rates

✔ Understand EMI obligations

✔ Keep emergency savings

✔ Avoid unnecessary debt

✔ Pay EMIs on time
""")

# ==========================================
# BORROWER RIGHTS
# ==========================================
elif page == "⚖ Borrower Rights":

    st.subheader("⚖ Borrower Awareness")

    st.success("""
✔ Know interest rates

✔ Understand fees and charges

✔ Receive repayment schedule

✔ Ask questions before signing

✔ Keep loan documents safe

✔ Understand all terms and conditions

✔ Maintain repayment discipline
""")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")

st.markdown("""
<div style='text-align:center;color:gray;'>

🏦 Smart Loan Advisor

Financial Eligibility & Risk Assessment Platform

</div>
""", unsafe_allow_html=True)