"""
Complete ML Pipeline for Loan Approval Prediction
Author: Akrati Mishra
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*70)
    print(" "*15 + "LOAN APPROVAL PREDICTION")
    print(" "*10 + "Machine Learning Classification Project")
    print("="*70)
    
    # Step 1: Generate sample data (since we don't have actual dataset)
    print("\n[1/5] Generating sample loan dataset...")
    df = generate_sample_data()
    print(f"Dataset created: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Step 2: Data Preprocessing
    print("\n[2/5] Data Preprocessing...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(df)
    
    # Step 3: Handle Imbalanced Data
    print("\n[3/5] Handling Imbalanced Data...")
    trainer = ModelTrainer()
    X_train_resampled, y_train_resampled = trainer.handle_imbalanced_data(
        X_train, y_train, method='smote'
    )
    
    # Step 4: Model Training
    print("\n[4/5] Training Models...")
    trained_models = trainer.train_all_models(X_train_resampled, y_train_resampled)
    
    # Step 5: Model Evaluation
    print("\n[5/5] Evaluating Models...")
    evaluator = ModelEvaluator()
    
    for model_name, model in trained_models.items():
        # Evaluate
        evaluator.evaluate_model(model, model_name, X_test, y_test)
        
        # Plot confusion matrix
        y_pred = model.predict(X_test)
        evaluator.plot_confusion_matrix(y_test, y_pred, model_name)
        
        # Plot ROC curve
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            evaluator.plot_roc_curve(y_test, y_pred_proba, model_name)
    
    # Compare all models
    evaluator.compare_models()
    evaluator.plot_model_comparison()
    
    # Save best model
    best_model_name = max(evaluator.results.items(), key=lambda x: x[1]['f1_score'])[0]
    best_model = trained_models[best_model_name]
    trainer.save_model(best_model, f'models/{best_model_name.replace(" ", "_")}_best.pkl')
    
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETED!")
    print("="*70)
    print(f"\n✅ Best Model: {best_model_name}")
    print(f"✅ F1-Score: {evaluator.results[best_model_name]['f1_score']:.4f}")
    print(f"✅ Model saved to: models/{best_model_name.replace(' ', '_')}_best.pkl")
    print("\n")

def generate_sample_data(n_samples=1000):
    """Generate sample loan dataset for demonstration"""
    np.random.seed(42)
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'ApplicantIncome': np.random.randint(1000, 10000, n_samples),
        'CoapplicantIncome': np.random.randint(0, 5000, n_samples),
        'LoanAmount': np.random.randint(50, 500, n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 120, 60], n_samples),
        'Credit_History': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable with some logic
    df['Loan_Status'] = 'N'
    
    # Higher chance of approval with good credit history and higher income
    approval_prob = (
        (df['Credit_History'] == 1) * 0.5 +
        (df['ApplicantIncome'] > 5000) * 0.2 +
        (df['Education'] == 'Graduate') * 0.15 +
        (df['Property_Area'] == 'Urban') * 0.1 +
        np.random.random(n_samples) * 0.05
    )
    
    df.loc[approval_prob > 0.5, 'Loan_Status'] = 'Y'
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'LoanAmount'] = np.nan
    
    return df

if __name__ == "__main__":
    main()