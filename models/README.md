# Models Directory

This directory stores trained machine learning models.

## Saved Models

After running the pipeline, you'll find:
- `Logistic_Regression_best.pkl`
- `Random_Forest_best.pkl`
- `XGBoost_best.pkl`
- `SVM_best.pkl`

## Loading Models

```python
import joblib

# Load a saved model
model = joblib.load('models/XGBoost_best.pkl')

# Make predictions
predictions = model.predict(X_new)
```

## Model Performance

Refer to the main README for detailed performance metrics of each model.