import streamlit as st
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Readmission Risk",page_icon = "🏥",layout="wide")
st.title("🏥 30-Day Readmission Risk Dashboard")

model=joblib.load('models/readmission_lgbm.pkl')
explainer=shap.TreeExplainer(model)

# Patient input form
with st.form("patient_form"):
    col1,col2,col3 = st.columns(3)
    with col1:
        age = st.number_input("Age",18,90,65)
        chronic = st.selectbox("Chronic condition",[0,1,2,3,4])
        los = st.number_input("Length od Stay (days)",1,45,5)
    with col2:
        dept = st.selectbox("Department",['cardiology','pulmonology','general_medicine'])
        admit_type = st.selectbox("Admission Type",['emergency','elective','urgent'])
        n_meds = st.number_input("Medications at discharge",1,12,4)
    with col3:
        creatinine = st.number_input("Creatinine",0.3,12.0,1.0)
        hemoglobin = st.number_input("Hemoglobin",5.0,20.0,12.5)
        discharge = st.selectbox("Discharge To",['home','home_health_service','rehab'])

    submitted=st.form_submit_button("Calculate Risk",type="primary")

if submitted:
    risk_score=0.35 #placeholder
    st.metric("Readmission Risk",f"{risk_score:.0%}",delta="HIGH RISK" if risk_score > 0.4 else "Low Risk")
