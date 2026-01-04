@echo off
echo 🚀 Starting Loan Approval Prediction System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

echo.

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

echo.

REM Install requirements
echo 📥 Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo ✅ All dependencies installed!
echo.

REM Run Streamlit app
echo 🌐 Starting web application...
echo 📱 The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py