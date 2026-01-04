"""
TROUBLESHOOTING GUIDE
Run this script to diagnose issues
"""

import sys
import subprocess

print("="*70)
print("  LOAN APPROVAL PREDICTION - TROUBLESHOOTING")
print("="*70)
print()

# Check Python version
print("1. Checking Python version...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("   ❌ ERROR: Python 3.8+ required")
    print("   Please upgrade Python")
else:
    print("   ✅ Python version OK")

print()

# Check pip
print("2. Checking pip...")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                          capture_output=True, text=True)
    print(f"   {result.stdout.strip()}")
    print("   ✅ pip is working")
except:
    print("   ❌ pip not found")

print()

# Try installing requirements
print("3. Installing/Checking requirements...")
print("   This may take a minute...")

try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", 
                   "--quiet"], check=True)
    print("   ✅ Requirements installed")
except Exception as e:
    print(f"   ❌ Error installing requirements: {e}")

print()

# Check critical imports
print("4. Checking critical packages...")

packages = {
    'streamlit': 'streamlit',
    'pandas': 'pandas', 
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'plotly': 'plotly'
}

all_ok = True
for import_name, package_name in packages.items():
    try:
        __import__(import_name)
        print(f"   ✅ {package_name}")
    except ImportError:
        print(f"   ❌ {package_name} - MISSING")
        print(f"      Install: pip install {package_name}")
        all_ok = False

print()

# Check if app.py exists
print("5. Checking app.py...")
try:
    with open('app.py', 'r') as f:
        content = f.read()
        if len(content) > 1000:
            print("   ✅ app.py exists and has content")
        else:
            print("   ⚠️  app.py seems too small")
except FileNotFoundError:
    print("   ❌ app.py not found!")
    print("      Make sure you're in the project directory")
    all_ok = False

print()

# Final verdict
print("="*70)
if all_ok:
    print("  ✅ SYSTEM READY!")
    print()
    print("  Run the app with:")
    print("  streamlit run app.py")
else:
    print("  ❌ ISSUES FOUND - Please fix the errors above")
    print()
    print("  Common fixes:")
    print("  1. pip install -r requirements.txt")
    print("  2. Make sure you're in the project directory")
    print("  3. Use Python 3.8 or higher")

print("="*70)