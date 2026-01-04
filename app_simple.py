"""
MINIMAL TEST VERSION - This WILL work!
Run: streamlit run app_simple.py
"""

import streamlit as st
import numpy as np

st.set_page_config(page_title="Loan Predictor", page_icon="💰")

st.title("💰 Loan Approval Predictor")
st.write("Fill in the details below:")

# Simple inputs
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
    loan = st.number_input("Loan Amount ($)", 10000, 500000, 100000)
    
with col2:
    credit = st.selectbox("Credit Score", ["Excellent", "Good", "Fair", "Poor"])
    employed = st.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed"])

if st.button("PREDICT", use_container_width=True):
    # Simple logic
    score = 0
    
    if income > 5000:
        score += 30
    if loan < 200000:
        score += 25
    if credit in ["Excellent", "Good"]:
        score += 35
    if employed == "Employed":
        score += 10
    
    # Add randomness
    score += np.random.randint(-5, 5)
    
    st.markdown("---")
    st.subheader("Results:")
    
    if score > 60:
        st.success(f"✅ APPROVED (Score: {score}/100)")
        st.balloons()
    else:
        st.error(f"❌ REJECTED (Score: {score}/100)")
    
    # Show details
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Approval Score", f"{score}%")
    with col2:
        st.metric("Risk Level", "Low" if score > 70 else "Medium" if score > 50 else "High")
    
    st.info("💡 This is a simplified demo. Real loan decisions involve many more factors.")

st.markdown("---")
st.caption("🤖 Simple ML Demo | Built with Streamlit")