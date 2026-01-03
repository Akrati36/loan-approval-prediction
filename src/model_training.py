import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def handle_imbalanced_data(self, X_train, y_train, method='smote'):
        """Handle imbalanced dataset"""
        print(f"\nHandling imbalanced data using {method.upper()}...")
        print(f"Original class distribution: {np.bincount(y_train)}")
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        elif method == 'combined':
            # Combine oversampling and undersampling
            over = SMOTE(sampling_strategy=0.5, random_state=42)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
            X_resampled, y_resampled = over.fit_resample(X_train, y_train)
            X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
        else:
            X_resampled, y_resampled = X_train, y_train
        
        print(f"Resampled class distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    
    def initialize_models(self):
        """Initialize ML models"""
        print("\nInitializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=1
            ),
            'SVM': SVC(
                random_state=42,
                probability=True,
                class_weight='balanced'
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_model(self, model_name, X_train, y_train, cv=5):
        """Train a single model with cross-validation"""
        print(f"\nTraining {model_name}...")
        
        model = self.models[model_name]
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        return model
    
    def hyperparameter_tuning(self, model_name, X_train, y_train):
        """Perform hyperparameter tuning"""
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return self.models[model_name]
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train, tune_hyperparameters=False):
        """Train all models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        self.initialize_models()
        trained_models = {}
        
        for model_name in self.models.keys():
            if tune_hyperparameters:
                trained_models[model_name] = self.hyperparameter_tuning(
                    model_name, X_train, y_train
                )
            else:
                trained_models[model_name] = self.train_model(
                    model_name, X_train, y_train
                )
        
        self.models = trained_models
        return trained_models
    
    def save_model(self, model, filepath):
        """Save trained model"""
        joblib.dump(model, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        model = joblib.load(filepath)
        print(f"\nModel loaded from {filepath}")
        return model

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    # trainer.initialize_models()
    # trained_models = trainer.train_all_models(X_train, y_train)