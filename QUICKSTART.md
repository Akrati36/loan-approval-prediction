# 🚀 Quick Start Guide

## Run Locally in 3 Steps

### Option 1: Using Quick Start Scripts

**On Mac/Linux:**
```bash
chmod +x start.sh
./start.sh
```

**On Windows:**
```bash
start.bat
```

The app will automatically:
- Create virtual environment
- Install all dependencies
- Launch the web app at http://localhost:8501

### Option 2: Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/Akrati36/loan-approval-prediction.git
cd loan-approval-prediction

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

## 🌐 Deploy to Cloud (FREE)

### Streamlit Cloud (Recommended - Takes 2 minutes!)

1. **Fork this repository** to your GitHub account

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Sign in** with GitHub

4. **Click** "New app"

5. **Fill in:**
   - Repository: `YourUsername/loan-approval-prediction`
   - Branch: `main`
   - Main file path: `app.py`

6. **Click** "Deploy"!

Your app will be live at: `https://your-app-name.streamlit.app`

### Other Deployment Options

#### Heroku
```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### Railway
1. Go to [railway.app](https://railway.app)
2. Connect GitHub repository
3. Deploy automatically

#### Render
1. Go to [render.com](https://render.com)
2. New Web Service
3. Connect repository
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run app.py`

## 📱 Using the App

1. **Fill in the form** in the left sidebar:
   - Personal information (gender, marital status, etc.)
   - Financial details (income, loan amount, etc.)
   - Property information

2. **Click** "PREDICT LOAN APPROVAL"

3. **View results:**
   - Approval/Rejection status
   - Probability percentage
   - Risk assessment
   - Feature importance
   - Personalized recommendations

## 🎯 Features

- ✅ **Real-time predictions** using trained ML model
- 📊 **Interactive visualizations** with Plotly
- 💡 **Smart recommendations** based on your profile
- 🔍 **Detailed analysis** of key factors
- 📈 **Probability gauge** for visual feedback
- 🎨 **Modern UI** with responsive design

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Model:** Random Forest Classifier (87% accuracy)
- **Visualization:** Plotly
- **Data Processing:** Pandas, NumPy, Scikit-learn

## 📊 Model Details

- **Algorithm:** Random Forest
- **Training Samples:** 2,000
- **Features:** 13 (including engineered features)
- **Accuracy:** 87%+
- **Handles:** Imbalanced data with class weights

## 🔒 Privacy & Security

- ✅ No data is stored or saved
- ✅ All processing happens in real-time
- ✅ No external API calls
- ✅ Completely secure and private

## 🐛 Troubleshooting

**Issue:** Port already in use
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

**Issue:** Module not found
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Issue:** Streamlit not found
```bash
# Solution: Install streamlit
pip install streamlit
```

## 📞 Support

- **Email:** akratimishra366@gmail.com
- **GitHub:** [@Akrati36](https://github.com/Akrati36)
- **Issues:** [Report here](https://github.com/Akrati36/loan-approval-prediction/issues)

## 🌟 Share Your Deployment

Once deployed, share your live demo:
- Add to your resume/portfolio
- Share on LinkedIn
- Include in job applications
- Show to recruiters

---

**Made with ❤️ by Akrati Mishra**