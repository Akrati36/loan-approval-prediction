# Loan Approval Prediction

A comprehensive Machine Learning project for predicting loan approval using classification algorithms with focus on handling imbalanced data.

## 🎯 Project Overview

This project demonstrates end-to-end ML workflow for loan approval prediction, covering:
- **Imbalanced Data Handling**: SMOTE, class weights, undersampling/oversampling
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, SVM
- **Feature Engineering**: Data preprocessing, encoding, scaling
- **Model Evaluation**: Confusion Matrix, Classification Report, ROC Curves

Perfect for **Data Analyst + ML Engineer** roles!

## 📊 Dataset Features

- **Applicant Information**: Gender, Marital Status, Dependents, Education
- **Financial Data**: Income, Loan Amount, Credit History, Property Area
- **Target Variable**: Loan Status (Approved/Rejected)

## 🛠️ Tech Stack

- **Python 3.8+**
- **Libraries**: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, matplotlib, seaborn
- **Jupyter Notebook** for interactive analysis

## 📁 Project Structure

```
loan-approval-prediction/
├── data/
│   ├── loan_data.csv
│   └── data_description.txt
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   └── 03_Model_Training.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── models/
│   └── saved_models/
├── requirements.txt
├── README.md
└── main.py
```

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Akrati36/loan-approval-prediction.git
cd loan-approval-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Project

```bash
# Run complete pipeline
python main.py

# Or use Jupyter notebooks for step-by-step analysis
jupyter notebook
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 82% | 80% | 85% | 82% | 0.87 |
| Random Forest | 85% | 83% | 88% | 85% | 0.91 |
| XGBoost | 87% | 85% | 89% | 87% | 0.93 |
| SVM | 81% | 79% | 84% | 81% | 0.86 |

## 🔍 Key Features

### 1. Exploratory Data Analysis
- Distribution analysis
- Correlation heatmaps
- Missing value analysis
- Target variable distribution

### 2. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Train-test split

### 3. Imbalanced Data Handling
- SMOTE (Synthetic Minority Over-sampling)
- Class weight adjustment
- Random undersampling
- Combination techniques

### 4. Model Training
- Multiple algorithms comparison
- Hyperparameter tuning (GridSearchCV)
- Cross-validation
- Feature importance analysis

### 5. Model Evaluation
- Confusion Matrix
- Classification Report
- ROC-AUC Curves
- Precision-Recall Curves

## 📊 Visualizations

- Feature correlation heatmap
- Distribution plots
- Confusion matrices
- ROC curves
- Feature importance charts

## 🎓 Learning Outcomes

- Handle imbalanced datasets effectively
- Implement multiple classification algorithms
- Evaluate models using appropriate metrics
- Feature engineering and selection
- Model comparison and selection
- Production-ready ML pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Akrati Mishra**
- GitHub: [@Akrati36](https://github.com/Akrati36)
- Email: akratimishra366@gmail.com

## 🌟 Acknowledgments

- Dataset inspired by real-world loan approval scenarios
- Built for Data Analyst and ML Engineer interview preparation

---

⭐ Star this repo if you find it helpful!