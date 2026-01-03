import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate a single model"""
        print(f"\n{'='*50}")
        print(f"EVALUATING: {model_name}")
        print('='*50)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print metrics
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"ROC-AUC:   {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
        
        return self.results[model_name]
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nConfusion Matrix saved for {model_name}")
    
    def plot_roc_curve(self, y_test, y_pred_proba, model_name):
        """Plot ROC curve"""
        if y_pred_proba is None:
            print(f"Cannot plot ROC curve for {model_name} - no probability predictions")
            return
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'roc_curve_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nROC Curve saved for {model_name}")
    
    def compare_models(self):
        """Compare all evaluated models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else 'N/A'
            })
        
        # Print comparison table
        print("\n{:<20} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
            'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'
        ))
        print("-" * 80)
        for data in comparison_data:
            print("{:<20} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
                data['Model'], data['Accuracy'], data['Precision'], 
                data['Recall'], data['F1-Score'], data['ROC-AUC']
            ))
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        print(f"\n🏆 Best Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
        
        return comparison_data
    
    def plot_model_comparison(self):
        """Plot model comparison chart"""
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            axes[idx].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nModel comparison chart saved!")

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    # evaluator.evaluate_model(model, 'Model Name', X_test, y_test)