import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/customer_predictions.csv')
    return df

@st.cache_data
def load_feature_importance():
    return pd.read_csv('reports/feature_importance.csv')

@st.cache_data
def load_metrics():
    with open('reports/final_model_metrics.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return joblib.load('models/final_model.joblib')

df = load_data()
feat_imp = load_feature_importance()
metrics = load_metrics()
model = load_model()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Executive Dashboard", "Customer Prediction", "Churn Drivers", "Customer Segmentation", "Business Recommendations"])

if page == "Executive Dashboard":
    st.title("Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(df)
    churn_rate = df['Churn'].mean() * 100
    retained = (df['Churn'] == 0).sum()
    churned = (df['Churn'] == 1).sum()
    high_risk = (df['Risk Segment'] == 'High Risk').sum()
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Churned Customers", f"{churned:,}")
    col4.metric("High-Risk Customers", f"{high_risk:,}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn Status")
        fig = px.pie(values=[retained, churned], names=['Retained', 'Churned'], hole=0.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Risk Segments")
        risk_counts = df['Risk Segment'].value_counts()
        fig2 = px.bar(x=risk_counts.index, y=risk_counts.values, color=risk_counts.index, color_discrete_map={'Low Risk':'green', 'Medium Risk':'orange', 'High Risk':'red'})
        fig2.update_layout(xaxis_title="Risk Level", yaxis_title="Number of Customers")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Customer Prediction":
    st.title("Predict Customer Churn")
    
    st.markdown("Enter customer details below to predict their churn likelihood.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior = st.selectbox("Senior Citizen", ['No', 'Yes'])
        partner = st.selectbox("Partner", ['No', 'Yes'])
        dependents = st.selectbox("Dependents", ['No', 'Yes'])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        
    with col2:
        phone = st.selectbox("Phone Service", ['Yes', 'No'])
        multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
        internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
        
    with col3:
        payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total = st.number_input("Total Charges ($)", min_value=0.0, value=monthly*tenure)
        online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    
    # Extra services hidden in expander to save space
    with st.expander("Additional Services"):
        c1, c2 = st.columns(2)
        with c1:
            online_backup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
            device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
        with c2:
            streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
            streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])

    if st.button("Predict Churn Risk"):
        tenure_group = '0-12'
        if 12 < tenure <= 24: tenure_group = '13-24'
        elif 24 < tenure <= 48: tenure_group = '25-48'
        elif tenure > 48: tenure_group = '49+'
            
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Senior Citizen': [senior],
            'Partner': [partner],
            'Dependents': [dependents],
            'Tenure Months': [tenure],
            'Phone Service': [phone],
            'Multiple Lines': [multiple_lines],
            'Internet Service': [internet],
            'Online Security': [online_security],
            'Online Backup': [online_backup],
            'Device Protection': [device_protection],
            'Tech Support': [tech_support],
            'Streaming TV': [streaming_tv],
            'Streaming Movies': [streaming_movies],
            'Contract': [contract],
            'Paperless Billing': [paperless],
            'Payment Method': [payment],
            'Monthly Charges': [monthly],
            'Total Charges': [total],
            'Tenure Group': [tenure_group]
        })
        
        prob = model.predict_proba(input_data)[0, 1]
        threshold = metrics['Threshold']
        pred = 1 if prob >= threshold else 0
        risk = "High Risk" if prob > 0.7 else "Medium Risk" if prob > 0.4 else "Low Risk"
        
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if pred == 1:
                st.error(f"⚠️ Likely to Churn (Probability: {prob:.1%})")
            else:
                st.success(f"✅ Likely to Stay (Probability: {prob:.1%})")
        with col_res2:
            if risk == "High Risk": st.error(f"Risk Segment: {risk}")
            elif risk == "Medium Risk": st.warning(f"Risk Segment: {risk}")
            else: st.success(f"Risk Segment: {risk}")

elif page == "Churn Drivers":
    st.title("Churn Drivers & Explainability")
    
    st.subheader("Top Predictors of Churn")
    fig = px.bar(feat_imp.head(15).sort_values('Importance'), x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Business Insights")
    st.markdown("""
    Based on the feature importance analysis:
    1. **Contract Type**: Month-to-month contracts are highly associated with churn compared to longer-term commitments.
    2. **Tenure**: Newer customers (especially 0-12 months) have a higher risk of leaving.
    3. **Internet Service (Fiber Optic)**: Customers with Fiber optic service show distinct churn patterns, potentially related to cost or service quality expectations.
    4. **Charges**: Higher monthly charges generally correlate with an increased likelihood of churn.
    """)

elif page == "Customer Segmentation":
    st.title("Customer Risk Segmentation")
    
    st.write(f"Showing predictions for {len(df):,} customers.")
    
    risk_filter = st.multiselect("Filter by Risk Segment", ['High Risk', 'Medium Risk', 'Low Risk'], default=['High Risk'])
    filtered_df = df[df['Risk Segment'].isin(risk_filter)]
    
    st.dataframe(filtered_df[['CustomerID', 'Tenure Months', 'Contract', 'Monthly Charges', 'Churn Probability', 'Risk Segment']].sort_values(by='Churn Probability', ascending=False).head(100))

elif page == "Business Recommendations":
    st.title("Business Recommendations & Strategy")
    
    st.markdown("""
    ### 1. Contract Conversion Strategy
    **Finding:** Customers on month-to-month contracts are the most likely to churn.
    **Action:** Offer targeted discounts or free service upgrades (e.g., 1 free month or free Tech Support) to month-to-month customers who agree to sign a 1-year contract.
    
    ### 2. Early Tenure Onboarding
    **Finding:** The risk of churn is highest in the first 12 months.
    **Action:** Implement a proactive 90-day onboarding program with regular check-ins to ensure new customers are getting value from their services.
    
    ### 3. Fiber Optic Customer Retention
    **Finding:** Fiber optic customers have distinct churn risk.
    **Action:** Investigate the specific reasons for Fiber optic churn (e.g., pricing vs. reliability). Offer targeted technical support and ensure competitive pricing for this high-value segment.
    
    ### 4. Service Bundling for Stickiness
    **Finding:** Customers without Tech Support or Online Security are more likely to leave.
    **Action:** Bundle these sticky services into standard packages at a slight discount to increase switching costs and improve customer satisfaction.
    
    ### 5. Prioritize High CLTV / High Risk Customers
    **Finding:** Not all churn has the same financial impact.
    **Action:** Combine the model's Churn Probability with CLTV to create a "Retention Priority" score. Allocate direct outreach resources (e.g., account manager calls) to High Risk + High CLTV customers first.
    """)
