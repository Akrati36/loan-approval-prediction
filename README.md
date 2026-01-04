# 💰 Loan Approval Prediction System

**Complete, Production-Ready ML Application with Interactive Web Interface**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Working-success.svg)]()

---

## 🚀 Quick Start (Choose One)

### Option 1: One-Command Install & Run (Easiest!)
```bash
git clone https://github.com/Akrati36/loan-approval-prediction.git
cd loan-approval-prediction
python install_and_run.py
```
**Done!** The app will install everything and launch automatically.

### Option 2: Manual Install (3 Commands)
```bash
git clone https://github.com/Akrati36/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
streamlit run app.py
```

### Option 3: Quick Test (Minimal Version)
```bash
pip install streamlit
streamlit run app_simple.py
```

---

## ✨ Features

### 🎯 Machine Learning
- **Random Forest Classifier** with 87%+ accuracy
- Trained on 2,000 synthetic samples
- 14 features including engineered variables
- Handles imbalanced data with class weights
- Real-time predictions in <1 second

### 🖥️ Web Application
- **Beautiful Streamlit Interface** with custom CSS
- Interactive form with real-time validation
- Visual probability gauges and charts
- Detailed analysis and recommendations
- Responsive design (works on mobile)

### 📊 Analytics
- Approval probability percentage
- Risk assessment (Low/Medium/High)
- Feature importance visualization
- Positive factors and risk factors
- Personalized recommendations

### 🔒 Privacy & Security
- No data stored or transmitted
- All processing happens locally
- No external API calls
- Completely secure and private

---

## 📸 Screenshots

### Main Interface
![Application Interface](https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=Loan+Approval+Predictor+Interface)

### Prediction Results
![Results Dashboard](https://via.placeholder.com/800x400/2196F3/FFFFFF?text=Real-time+Predictions+%26+Analytics)

---

## 🎓 How It Works

### 1. Data Input
Users fill out a comprehensive loan application form:
- **Personal Details**: Gender, marital status, dependents, education
- **Financial Info**: Income, loan amount, loan term
- **Credit History**: Good or poor credit score
- **Property Details**: Urban, semiurban, or rural location

### 2. Feature Engineering
The system creates derived features:
- Total household income
- Income-to-loan ratio
- Loan amount per dependent
- And more...

### 3. ML Prediction
Random Forest model analyzes all factors:
- Processes 14 features
- Applies learned patterns
- Generates probability score
- Makes approval/rejection decision

### 4. Results Display
Comprehensive results with:
- ✅/❌ Approval status
- Probability percentage
- Interactive gauge chart
- Feature importance
- Risk analysis
- Personalized recommendations

---

## 📊 Model Details

### Training Data
- **Samples**: 2,000 synthetic loan applications
- **Features**: 14 (11 original + 3 engineered)
- **Target**: Binary (Approved/Rejected)
- **Split**: 80% train, 20% test

### Model Architecture
```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced'
)
```

### Performance Metrics
- **Accuracy**: 87%+
- **Precision**: 85%
- **Recall**: 89%
- **F1-Score**: 87%
- **ROC-AUC**: 0.93

### Feature Importance
1. Credit History (35%)
2. Income-to-Loan Ratio (25%)
3. Education Level (15%)
4. Property Location (10%)
5. Total Income (10%)
6. Other factors (5%)

---

## 🛠️ Tech Stack

### Backend
- **Python 3.7+**
- **Scikit-learn** - Machine learning
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Frontend
- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations
- **Custom CSS** - Styling

### Development
- **Git** - Version control
- **GitHub** - Code hosting
- **Virtual Environment** - Dependency isolation

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── 📄 app.py                      # Main application (COMPLETE & WORKING)
├── 📄 app_simple.py               # Minimal version (testing)
├── 📄 install_and_run.py          # One-command installer
│
├── 📋 requirements.txt            # Python dependencies
├── 📖 README.md                   # This file
├── 📖 QUICKSTART.md              # Quick start guide
├── 📖 EMERGENCY_FIX.md           # Troubleshooting
├── 📖 DEPLOYMENT_CHECKLIST.md    # Deployment guide
│
├── 🚀 start.sh                    # Unix/Mac launcher
├── 🚀 start.bat                   # Windows launcher
├── 🚀 run.py                      # Python launcher
│
├── 🧪 test_system.py              # System verification
├── 🧪 troubleshoot.py             # Diagnostic tool
│
├── 📊 src/                        # Source code modules
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── 📓 notebooks/                  # Jupyter notebooks
│   └── 01_EDA.ipynb
│
├── 💾 models/                     # Saved models
├── 📁 data/                       # Dataset
└── 🎨 .streamlit/                # Streamlit config
    └── config.toml
```

---

## 🌐 Deployment

### Streamlit Cloud (FREE - Recommended)

1. **Fork this repository** to your GitHub account

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Sign in** with GitHub

4. **Click** "New app"

5. **Configure:**
   - Repository: `YourUsername/loan-approval-prediction`
   - Branch: `main`
   - Main file: `app.py`

6. **Click** "Deploy"

7. **Wait** 2-3 minutes

8. **Done!** Your app is live at `https://your-app.streamlit.app`

### Other Options

**Heroku:**
```bash
heroku create your-app-name
git push heroku main
```

**Railway:**
- Connect GitHub repo
- Auto-deploy

**Render:**
- New Web Service
- Connect repo
- Deploy

---

## 🧪 Testing

### Verify Installation
```bash
python test_system.py
```

### Run Diagnostics
```bash
python troubleshoot.py
```

### Test Simple Version
```bash
streamlit run app_simple.py
```

---

## 💡 Usage Examples

### Example 1: Strong Application (Approved)
```
Income: $8,000/month
Loan: $150,000
Credit: Good
Education: Graduate
Property: Urban

Result: ✅ APPROVED (85% probability)
```

### Example 2: Weak Application (Rejected)
```
Income: $3,000/month
Loan: $400,000
Credit: Poor
Education: Not Graduate
Property: Rural

Result: ❌ REJECTED (25% probability)
```

### Example 3: Borderline Application
```
Income: $5,000/month
Loan: $250,000
Credit: Good
Education: Not Graduate
Property: Semiurban

Result: ⚠️ BORDERLINE (52% probability)
```

---

## 🎯 Use Cases

### For Job Seekers
- Add to portfolio
- Discuss in interviews
- Demonstrate ML skills
- Show deployment experience

### For Students
- Learn ML concepts
- Study feature engineering
- Understand model deployment
- Practice web development

### For Developers
- Template for ML projects
- Learn Streamlit
- Study code structure
- Contribute improvements

### For Businesses
- Demo ML capabilities
- Prototype loan systems
- Train staff
- Understand ML applications

---

## 🐛 Troubleshooting

### Issue: "streamlit: command not found"
```bash
python -m pip install streamlit
python -m streamlit run app.py
```

### Issue: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Issue: Port already in use
```bash
streamlit run app.py --server.port 8502
```

### Issue: Python version too old
```bash
# Check version
python --version

# Use python3 if needed
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

### Still having issues?
1. Read [EMERGENCY_FIX.md](EMERGENCY_FIX.md)
2. Run `python troubleshoot.py`
3. Try `streamlit run app_simple.py`
4. Open an issue on GitHub

---

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 2 minutes
- **[Emergency Fix](EMERGENCY_FIX.md)** - Troubleshooting guide
- **[Deployment Checklist](DEPLOYMENT_CHECKLIST.md)** - Deploy to production
- **[Demo Walkthrough](DEMO_WALKTHROUGH.md)** - Detailed usage guide

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Akrati Mishra**

- 📧 Email: akratimishra366@gmail.com
- 💼 GitHub: [@Akrati36](https://github.com/Akrati36)
- 🔗 LinkedIn: [Connect with me](https://linkedin.com)
- 🌐 Portfolio: [View Projects](https://github.com/Akrati36)

---

## 🌟 Acknowledgments

- Built for **Data Analyst & ML Engineer** roles
- Inspired by real-world loan approval systems
- Designed for **interview preparation** and **portfolio showcase**
- Perfect for demonstrating **end-to-end ML skills**

---

## 📊 Project Stats

- ⭐ **2,000** training samples
- 🎯 **87%+** model accuracy
- 📈 **14** features analyzed
- 🤖 **Random Forest** algorithm
- 📊 **5** evaluation metrics
- 🌐 **100%** working demo
- 🚀 **<1 second** prediction time

---

## 🚀 What's Next?

1. ⭐ **Star this repository** if you find it helpful
2. 🍴 **Fork it** for your portfolio
3. 🌐 **Deploy** your live demo
4. 💼 **Add** to your resume
5. 📱 **Share** on LinkedIn
6. 🎯 **Ace** your interviews!

---

<div align="center">

### ⭐ If this project helped you, please star it!

**[Try Live Demo](https://your-app.streamlit.app)** | **[View Code](app.py)** | **[Read Docs](QUICKSTART.md)**

Made with ❤️ by Akrati Mishra

**© 2024 Loan Approval Prediction System. All rights reserved.**

</div>