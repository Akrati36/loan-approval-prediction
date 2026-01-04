#!/usr/bin/env python3
"""
Simple run script - Just run this!
"""

import subprocess
import sys

print("🚀 Starting Loan Approval Predictor...")
print()

# Install requirements
print("📦 Installing requirements...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                "streamlit", "pandas", "numpy", "scikit-learn", "plotly"])

print("✅ Requirements installed!")
print()

# Run app
print("🌐 Launching app...")
print("📱 Opening http://localhost:8501")
print()

subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])