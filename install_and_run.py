#!/usr/bin/env python3
"""
ONE-COMMAND INSTALLER AND LAUNCHER
Just run: python install_and_run.py
"""

import subprocess
import sys
import os

def print_header():
    print("=" * 70)
    print("  LOAN APPROVAL PREDICTION SYSTEM - AUTO INSTALLER")
    print("=" * 70)
    print()

def check_python():
    print("✓ Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro} detected")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ❌ ERROR: Python 3.7+ required")
        return False
    
    print("  ✅ Python version OK")
    return True

def install_packages():
    print("\n📦 Installing required packages...")
    print("  This may take 1-2 minutes...\n")
    
    packages = [
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly"
    ]
    
    for package in packages:
        print(f"  Installing {package}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                check=True,
                capture_output=True
            )
            print(f"  ✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  {package} installation had issues (may already be installed)")
    
    print("\n✅ All packages installed!")

def launch_app():
    print("\n🚀 Launching application...")
    print("  The app will open in your browser at http://localhost:8501")
    print("  Press Ctrl+C to stop the server\n")
    print("=" * 70)
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped. Thanks for using!")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        print("\nTry running manually:")
        print("  streamlit run app.py")

def main():
    print_header()
    
    if not check_python():
        sys.exit(1)
    
    install_packages()
    launch_app()

if __name__ == "__main__":
    main()