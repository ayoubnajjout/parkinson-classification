# Parkinson's Disease Detection using ML Models

## Overview
Custom implementation of machine learning models (Logistic Regression, Random Forest, LightGBM) for detecting Parkinson's disease through voice analysis. The project focuses on feature selection techniques and model optimization.

## Project Structure

pd-detection/ ├── mini-projet-ml.ipynb # Main analysis notebook ├── data/ │ └── pd_speech_features.csv # Voice dataset └── README.md

## Technical Implementation

### Data Pipeline
```python
# filepath: mini-projet-ml.ipynb
# 1. Data Loading & Preprocessing
df = pd.read_csv('data/pd_speech_features.csv')
X, y = preprocess_data(df)

# 2. Feature Selection
X_selected = feature_selection(X, y, method='combined')

# 3. Model Training
models = {
    'logistic': LogisticRegression(),
    'random_forest': RandomForest(),
    'lightgbm': LightGBM()
}

Model Architecture

graph TD
    A[Voice Data Input] --> B[Preprocessing]
    B --> C[Feature Selection]
    C --> D[Model Training]
    D --> E[Model Evaluation]
    
    C --> |Filter| F[Correlation]
    C --> |Embedded| G[Lasso]
    C --> |Wrapper| H[Backward]
    
    F --> I[Final Features]
    G --> I
    H --> I
    
    I --> J[Train Models]
    J --> K[LR/RF/LGBM]
    K --> L[Results]

Setup Instructions

# Windows setup
python -m venv pd-env
.\pd-env\Scripts\activate
pip install -r requirements.txt

Dependencies

# filepath: requirements.txt
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
matplotlib==3.4.2
seaborn==0.11.1
jupyter==1.0.0

Results Summary

Model	Accuracy	F1-Score
LR	86.2%	0.85
RF	84.7%	0.83
LGBM	85.9%	0.84

Key findings:

Feature selection improved model performance
Logistic Regression performed best overall
Combined feature selection approach most effective
Future Improvements
Deep learning implementation
Real-time prediction API
Web interface deployment
Additional voice features
Hyperparameter optimization
