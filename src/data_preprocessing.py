import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self, filepath):
        """Load the loan dataset"""
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("\nHandling missing values...")
        
        # For numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        print("Missing values handled!")
        return df
    
    def encode_categorical(self, df, target_col='Loan_Status'):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Encode target variable
        if target_col in df.columns:
            le_target = LabelEncoder()
            df[target_col] = le_target.fit_transform(df[target_col])
            self.label_encoders[target_col] = le_target
        
        print("Categorical encoding completed!")
        return df
    
    def feature_engineering(self, df):
        """Create new features"""
        print("\nPerforming feature engineering...")
        
        # Total income
        if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
            df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Income to loan ratio
        if 'TotalIncome' in df.columns and 'LoanAmount' in df.columns:
            df['IncomeToLoanRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
        
        # Loan amount per term
        if 'LoanAmount' in df.columns and 'Loan_Amount_Term' in df.columns:
            df['LoanAmountPerTerm'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)
        
        print("Feature engineering completed!")
        return df
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        print("\nScaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Feature scaling completed!")
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self, df, target_col='Loan_Status', test_size=0.2, random_state=42):
        """Complete data preparation pipeline"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df, target_col)
        
        # Split features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Class distribution in training set:")
        print(y_train.value_counts(normalize=True))
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    # df = preprocessor.load_data('data/loan_data.csv')
    # X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(df)