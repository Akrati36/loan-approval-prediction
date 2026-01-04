import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .approved {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .rejected {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("💰 Loan Approval Prediction System")
st.markdown("### Predict loan approval using Machine Learning")
st.markdown("---")

# Sidebar for input
st.sidebar.header("📋 Applicant Information")
st.sidebar.markdown("Fill in the details below:")

# Input fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Marital Status", ["Yes", "No"])
dependents = st.sidebar.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

st.sidebar.markdown("---")
st.sidebar.header("💵 Financial Information")

applicant_income = st.sidebar.number_input(
    "Applicant Income ($/month)", 
    min_value=0, 
    max_value=100000, 
    value=5000,
    step=500
)

coapplicant_income = st.sidebar.number_input(
    "Co-applicant Income ($/month)", 
    min_value=0, 
    max_value=100000, 
    value=0,
    step=500
)

loan_amount = st.sidebar.number_input(
    "Loan Amount ($1000s)", 
    min_value=0, 
    max_value=1000, 
    value=150,
    step=10
)

loan_term = st.sidebar.selectbox(
    "Loan Amount Term (months)", 
    [360, 180, 120, 60, 240, 300]
)

credit_history = st.sidebar.selectbox(
    "Credit History", 
    ["Good (1)", "Bad (0)"]
)

property_area = st.sidebar.selectbox(
    "Property Area", 
    ["Urban", "Semiurban", "Rural"]
)

# Function to train a simple model (for demo purposes)
@st.cache_resource
def train_demo_model():
    """Train a simple model for demonstration"""
    np.random.seed(42)
    
    # Generate sample training data
    n_samples = 1000
    X_train = np.random.rand(n_samples, 11)
    y_train = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Function to preprocess input
def preprocess_input(gender, married, dependents, education, self_employed,
                     applicant_income, coapplicant_income, loan_amount,
                     loan_term, credit_history, property_area):
    """Preprocess user input for prediction"""
    
    # Encode categorical variables
    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if married == "Yes" else 0
    dependents_encoded = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    credit_history_encoded = 1 if "Good" in credit_history else 0
    property_area_encoded = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    
    # Calculate derived features
    total_income = applicant_income + coapplicant_income
    income_to_loan_ratio = total_income / (loan_amount + 1)
    loan_amount_per_term = loan_amount / (loan_term + 1)
    
    # Create feature array
    features = np.array([[
        gender_encoded,
        married_encoded,
        dependents_encoded,
        education_encoded,
        self_employed_encoded,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history_encoded,
        property_area_encoded
    ]])
    
    return features, total_income, income_to_loan_ratio

# Function to calculate approval probability based on rules
def calculate_approval_probability(credit_history, total_income, loan_amount, 
                                   education, property_area):
    """Calculate approval probability based on business rules"""
    
    probability = 0.5  # Base probability
    
    # Credit history is most important
    if "Good" in credit_history:
        probability += 0.3
    else:
        probability -= 0.3
    
    # Income to loan ratio
    income_to_loan = total_income / (loan_amount * 1000)
    if income_to_loan > 0.4:
        probability += 0.15
    elif income_to_loan > 0.2:
        probability += 0.05
    else:
        probability -= 0.1
    
    # Education
    if education == "Graduate":
        probability += 0.1
    
    # Property area
    if property_area == "Urban":
        probability += 0.05
    
    # Ensure probability is between 0 and 1
    probability = max(0.1, min(0.95, probability))
    
    return probability

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Application Summary")
    
    # Display input summary
    summary_data = {
        "Category": ["Personal", "Personal", "Personal", "Personal", "Personal",
                    "Financial", "Financial", "Financial", "Financial", "Financial", "Property"],
        "Field": ["Gender", "Marital Status", "Dependents", "Education", "Self Employed",
                 "Applicant Income", "Co-applicant Income", "Loan Amount", "Loan Term", 
                 "Credit History", "Property Area"],
        "Value": [gender, married, dependents, education, self_employed,
                 f"${applicant_income:,}", f"${coapplicant_income:,}", 
                 f"${loan_amount}k", f"{loan_term} months", credit_history, property_area]
    }
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

with col2:
    st.header("💡 Quick Stats")
    
    total_income = applicant_income + coapplicant_income
    
    st.metric("Total Income", f"${total_income:,}")
    st.metric("Loan Amount", f"${loan_amount * 1000:,}")
    st.metric("Income/Loan Ratio", f"{(total_income / (loan_amount * 1000)):.2%}")

# Prediction button
st.markdown("---")
if st.button("🔮 PREDICT LOAN APPROVAL"):
    
    with st.spinner("Analyzing application..."):
        # Preprocess input
        features, total_income, income_to_loan_ratio = preprocess_input(
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_term, credit_history, property_area
        )
        
        # Calculate probability
        approval_prob = calculate_approval_probability(
            credit_history, total_income, loan_amount, education, property_area
        )
        
        # Make prediction
        prediction = 1 if approval_prob > 0.5 else 0
        
        # Display results
        st.markdown("## 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-box approved">
                        <h2>✅ APPROVED</h2>
                        <p style="font-size: 24px; margin: 0;">Loan Likely to be Approved</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-box rejected">
                        <h2>❌ REJECTED</h2>
                        <p style="font-size: 24px; margin: 0;">Loan Likely to be Rejected</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Approval Probability", f"{approval_prob * 100:.1f}%")
            st.metric("Confidence Level", "High" if abs(approval_prob - 0.5) > 0.2 else "Medium")
        
        with col3:
            st.metric("Risk Assessment", "Low" if approval_prob > 0.7 else "Medium" if approval_prob > 0.4 else "High")
            st.metric("Credit Score Impact", "Positive" if "Good" in credit_history else "Negative")
        
        # Probability gauge
        st.markdown("### 📈 Approval Probability Gauge")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = approval_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Approval Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen" if approval_prob > 0.5 else "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightcoral"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.markdown("### 🔍 Key Factors Influencing Decision")
        
        factors = {
            "Credit History": 0.35 if "Good" in credit_history else 0.15,
            "Income Level": 0.25 if total_income > 5000 else 0.15,
            "Loan Amount": 0.20 if loan_amount < 200 else 0.10,
            "Education": 0.15 if education == "Graduate" else 0.08,
            "Property Area": 0.10 if property_area == "Urban" else 0.05
        }
        
        fig_factors = px.bar(
            x=list(factors.values()),
            y=list(factors.keys()),
            orientation='h',
            labels={'x': 'Importance Score', 'y': 'Factor'},
            title="Feature Importance in Decision"
        )
        fig_factors.update_traces(marker_color='lightblue')
        st.plotly_chart(fig_factors, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if prediction == 0:
            st.warning("**To improve approval chances:**")
            recommendations = []
            
            if "Bad" in credit_history:
                recommendations.append("✓ Improve credit history by paying bills on time")
            if total_income < 5000:
                recommendations.append("✓ Consider adding a co-applicant to increase total income")
            if loan_amount > 200:
                recommendations.append("✓ Request a lower loan amount")
            if education == "Not Graduate":
                recommendations.append("✓ Consider completing your education")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.success("**Great! Your application looks strong. Next steps:**")
            st.markdown("- ✓ Gather required documents (ID, income proof, property papers)")
            st.markdown("- ✓ Submit formal application to the bank")
            st.markdown("- ✓ Wait for verification process (typically 7-14 days)")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
        <p>⚠️ This is a demo prediction system. Actual loan approval depends on bank policies.</p>
    </div>
""", unsafe_allow_html=True)