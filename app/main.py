import streamlit as st
from prediction_helper import predict

st.set_page_config(page_title="Credit Risk Modelling", page_icon="📈", layout="wide")
st.title("Predict Your Credit Risk Now")
st.markdown("#")

rows = [st.columns(4) for _ in range(3)]

with rows[0][0]: age = st.number_input('**Age**', min_value=18, step=1, max_value=100, value=28)
with rows[0][1]: income = st.number_input('**Income**', min_value=0, value=1200000)
with rows[0][2]: loan_amount = st.number_input('**Loan Amount**', min_value=0, value=2560000)
with rows[0][3]: loan_tenure_months = st.number_input('**Loan Tenure (months)**', min_value=0, step=1, value=36)

loan_to_income_ratio = loan_amount / income if income > 0 else 0
with rows[1][0]: st.number_input('**Loan to Income Ratio (Auto-Calculated, Read-Only)**', value=loan_to_income_ratio, format="%.2f", disabled=True)
with rows[1][1]: avg_dpd_per_delinquency = st.number_input('**Avg DPD (Days Past Due)**', min_value=0, value=20)
with rows[1][2]: delinquency_ratio = st.number_input('**Delinquency Ratio**', min_value=0, max_value=100, step=1, value=30)
with rows[1][3]: credit_utilization_ratio = st.number_input('**Credit Utilization Ratio**', min_value=0, max_value=100, step=1, value=30)

with rows[2][0]: num_open_accounts = st.number_input('**Open Loan Accounts**', min_value=1, max_value=4, step=1, value=2)
with rows[2][1]: residence_type = st.selectbox('**Residence Type**', ['Owned', 'Rented', 'Mortgage'])
with rows[2][2]: loan_purpose = st.selectbox('**Loan Purpose**', ['Education', 'Home', 'Auto', 'Personal'])
with rows[2][3]: loan_type = st.selectbox('**Loan Type**', ['Unsecured', 'Secured'])


if st.button('Predict Risk'):
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                                                delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)
    st.success(f"""
    **Default Probability:** {probability:.2%}\n
    **Credit Score:** {credit_score}\n
    **Rating:** {rating}
    """)
