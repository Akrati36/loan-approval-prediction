import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Generate and train model
@st.cache_resource
def load_model():
    """Generate data and train model"""
    try:
        np.random.seed(42)
        n = 2000
        
        # Generate data
        data = {
            'Gender': np.random.choice([0, 1], n),
            'Married': np.random.choice([0, 1], n),
            'Dependents': np.random.choice([0, 1, 2, 3], n),
            'Education': np.random.choice([0, 1], n),
            'Self_Employed': np.random.choice([0, 1], n),
            'ApplicantIncome': np.random.randint(1000, 15000, n),
            'CoapplicantIncome': np.random.randint(0, 8000, n),
            'LoanAmount': np.random.randint(50, 600, n),
            'Loan_Amount_Term': np.random.choice([360, 180, 120, 240], n),
            'Credit_History': np.random.choice([0, 1], n, p=[0.15, 0.85]),
            'Property_Area': np.random.choice([0, 1, 2], n),
        }
        
        df = pd.DataFrame(data)
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['IncomeToLoanRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
        
        # Create target
        score = (
            df['Credit_History'] * 0.4 +
            (df['IncomeToLoanRatio'] > 30).astype(int) * 0.25 +
            df['Education'] * 0.15 +
            (df['Property_Area'] == 2).astype(int) * 0.1 +
            (df['TotalIncome'] > 6000).astype(int) * 0.1
        )
        score += np.random.uniform(-0.15, 0.15, n)
        df['Loan_Status'] = (score > 0.5).astype(int)
        
        # Prepare features
        features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                   'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                   'Loan_Amount_Term', 'Credit_History', 'Property_Area',
                   'TotalIncome', 'IncomeToLoanRatio']
        
        X = df[features]
        y = df['Loan_Status']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        accuracy = model.score(X_test_scaled, y_test)
        
        return model, scaler, accuracy, features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, 0, []

# Load model
model, scaler, accuracy, feature_names = load_model()

# Title
st.title("💰 Loan Approval Prediction System")
st.markdown(f"### AI-Powered Loan Decision Support | Model Accuracy: {accuracy*100:.1f}%")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("📋 Application Details")

# Personal info
st.sidebar.subheader("👤 Personal Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Marital Status", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])

# Financial info
st.sidebar.subheader("💵 Financial Information")
applicant_income = st.sidebar.slider("Applicant Income ($/month)", 1000, 15000, 5000, 500)
coapplicant_income = st.sidebar.slider("Co-applicant Income ($/month)", 0, 8000, 0, 500)
loan_amount = st.sidebar.slider("Loan Amount ($1000s)", 50, 600, 150, 10)
loan_term = st.sidebar.selectbox("Loan Term (months)", [360, 180, 120, 240, 300])
credit_history = st.sidebar.selectbox("Credit History", ["Good (1.0)", "Poor (0.0)"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Main area - Summary
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Application Summary")
    summary = pd.DataFrame({
        "Field": ["Gender", "Married", "Dependents", "Education", "Self Employed",
                 "Applicant Income", "Co-applicant Income", "Loan Amount", 
                 "Loan Term", "Credit History", "Property Area"],
        "Value": [gender, married, dependents, education, self_employed,
                 f"${applicant_income:,}", f"${coapplicant_income:,}", 
                 f"${loan_amount}k", f"{loan_term} months", credit_history, property_area]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

with col2:
    st.header("💡 Quick Stats")
    total_income = applicant_income + coapplicant_income
    st.metric("Total Income", f"${total_income:,}")
    st.metric("Loan Amount", f"${loan_amount * 1000:,}")
    if loan_amount > 0:
        ratio = (total_income / (loan_amount * 1000)) * 100
        st.metric("Income/Loan %", f"{ratio:.1f}%")

# Predict button
st.markdown("---")

if st.button("🔮 PREDICT LOAN APPROVAL"):
    if model is None:
        st.error("❌ Model not loaded. Please refresh the page.")
    else:
        with st.spinner("Analyzing..."):
            # Encode inputs
            gender_enc = 1 if gender == "Male" else 0
            married_enc = 1 if married == "Yes" else 0
            dependents_enc = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
            education_enc = 1 if education == "Graduate" else 0
            self_employed_enc = 1 if self_employed == "Yes" else 0
            credit_enc = 1 if "Good" in credit_history else 0
            property_enc = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
            
            total_income = applicant_income + coapplicant_income
            income_ratio = total_income / (loan_amount + 1)
            
            # Create feature array
            features = np.array([[
                gender_enc, married_enc, dependents_enc, education_enc, 
                self_employed_enc, applicant_income, coapplicant_income,
                loan_amount, loan_term, credit_enc, property_enc,
                total_income, income_ratio
            ]])
            
            # Scale and predict
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            approval_prob = proba[1]
            
            # Display results
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("### ✅ APPROVED")
                    st.markdown("**Loan Likely to be Approved**")
                else:
                    st.error("### ❌ REJECTED")
                    st.markdown("**Loan Likely to be Rejected**")
            
            with col2:
                st.metric("Approval Probability", f"{approval_prob*100:.1f}%")
                confidence = "High" if abs(approval_prob - 0.5) > 0.25 else "Medium"
                st.metric("Confidence", confidence)
            
            with col3:
                risk = "Low" if approval_prob > 0.7 else "Medium" if approval_prob > 0.4 else "High"
                st.metric("Risk Level", risk)
                st.metric("Score", f"{approval_prob*100:.0f}/100")
            
            # Probability gauge
            st.markdown("### 📈 Approval Probability")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=approval_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green" if approval_prob > 0.5 else "red"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightcoral"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': 50}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            st.markdown("### 📋 Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Positive Factors")
                if credit_enc == 1:
                    st.success("✓ Good credit history")
                if total_income > 6000:
                    st.success("✓ Strong income")
                if education_enc == 1:
                    st.success("✓ Graduate education")
                if property_enc == 2:
                    st.success("✓ Urban property")
            
            with col2:
                st.markdown("#### ⚠️ Risk Factors")
                if credit_enc == 0:
                    st.warning("⚠ Poor credit history")
                if total_income < 4000:
                    st.warning("⚠ Low income")
                if loan_amount > 400:
                    st.warning("⚠ High loan amount")
                if income_ratio < 20:
                    st.warning("⚠ Low income/loan ratio")
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if prediction == 0:
                st.error("**To improve approval chances:**")
                if credit_enc == 0:
                    st.markdown("- Improve credit score")
                if total_income < 5000:
                    st.markdown("- Increase income or add co-applicant")
                if loan_amount > 300:
                    st.markdown("- Request lower loan amount")
            else:
                st.success("**Next Steps:**")
                st.markdown("- Gather required documents")
                st.markdown("- Submit formal application")
                st.markdown("- Wait for verification (7-14 days)")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
        <p>⚠️ Demo system for educational purposes</p>
    </div>
""", unsafe_allow_html=True)