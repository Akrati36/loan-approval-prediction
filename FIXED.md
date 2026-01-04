# ✅ FIXED - SUPER SIMPLE INSTRUCTIONS

## Run in 1 Command!

```bash
python run.py
```

That's it! The script will:
1. Install everything needed
2. Launch the app
3. Open in your browser

## Alternative: 3 Commands

```bash
pip install streamlit pandas numpy scikit-learn plotly
streamlit run app.py
```

## What I Fixed:

✅ Simplified app.py (removed complex code)
✅ Minimal requirements (only 5 packages)
✅ Added run.py (one-command start)
✅ Better error handling
✅ Faster loading

## If You Still Have Issues:

**Issue: "python: command not found"**
```bash
python3 run.py
```

**Issue: "pip: command not found"**
```bash
python -m pip install streamlit pandas numpy scikit-learn plotly
python -m streamlit run app.py
```

**Issue: "Permission denied"**
```bash
chmod +x run.py
./run.py
```

**Issue: Port already in use**
```bash
streamlit run app.py --server.port 8502
```

## Test It Works:

1. Clone repo:
```bash
git clone https://github.com/Akrati36/loan-approval-prediction.git
cd loan-approval-prediction
```

2. Run:
```bash
python run.py
```

3. Open browser: http://localhost:8501

## What You'll See:

- Form on left sidebar
- Fill in loan details
- Click "PREDICT LOAN APPROVAL"
- See results instantly!

## Still Not Working?

Tell me:
1. What command did you run?
2. What error message appeared?
3. Windows/Mac/Linux?

I'll fix it immediately! 🔧