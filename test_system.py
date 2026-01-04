"""
Test script to verify the loan approval prediction system
Run this to ensure everything is working correctly
"""

import sys
import subprocess

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Testing package imports...")
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'streamlit',
        'plotly',
        'imblearn'
    ]
    
    failed = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT FOUND")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages installed correctly!")
    return True

def test_model_training():
    """Test if model can be trained"""
    print("\n🧪 Testing model training...")
    
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Generate sample data
        np.random.seed(42)
        X = np.random.rand(100, 10)
        y = np.random.choice([0, 1], 100)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test prediction
        accuracy = model.score(X_test, y_test)
        
        print(f"  ✅ Model trained successfully!")
        print(f"  ✅ Test accuracy: {accuracy*100:.2f}%")
        return True
        
    except Exception as e:
        print(f"  ❌ Model training failed: {e}")
        return False

def test_streamlit():
    """Test if Streamlit is working"""
    print("\n🧪 Testing Streamlit...")
    
    try:
        import streamlit as st
        print("  ✅ Streamlit imported successfully!")
        print(f"  ✅ Streamlit version: {st.__version__}")
        return True
    except Exception as e:
        print(f"  ❌ Streamlit test failed: {e}")
        return False

def test_app_file():
    """Test if app.py exists and is valid"""
    print("\n🧪 Testing app.py file...")
    
    try:
        with open('app.py', 'r') as f:
            content = f.read()
            
        if 'streamlit' in content and 'RandomForestClassifier' in content:
            print("  ✅ app.py found and looks valid!")
            return True
        else:
            print("  ⚠️  app.py found but may be incomplete")
            return False
            
    except FileNotFoundError:
        print("  ❌ app.py not found!")
        return False
    except Exception as e:
        print(f"  ❌ Error reading app.py: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("  LOAN APPROVAL PREDICTION - SYSTEM TEST")
    print("="*60)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Model Training", test_model_training),
        ("Streamlit", test_streamlit),
        ("App File", test_app_file)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print()
    print(f"  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready to run.")
        print("\n📝 Next steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Open browser at: http://localhost:8501")
        print("  3. Start predicting loan approvals!")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("  Run: pip install -r requirements.txt")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()