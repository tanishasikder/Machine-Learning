Health Insurance Cost Prediction
This project predicts individual insurance charges using features like age, BMI, number of children, sex, smoking status, and region. It uses multiple machine learning models for comparison and allows user input for real-time prediction.

Dataset
Source: Kaggle - Medical Cost Personal Dataset

Features:
    age: Age of primary beneficiary
    sex: Gender (male/female)
    bmi: Body mass index
    children: Number of dependents
    smoker: Smoker or non-smoker
    region: Geographic region
    charges: Medical insurance cost (target variable)

Tools & Libraries Used
    Python
    pandas, numpy, matplotlib
    scikit-learn for preprocessing & ML models
    xgboost for gradient boosting
    mpl_toolkits for 3D visualization

Project Features
🔹 Preprocessing
        One-hot encoding of categorical features (sex, smoker, region)
        Normalization (for SVR)
        Custom helper functions to clean and prepare user input

🔹 Models Trained
        Linear Regression
        Decision Tree Regressor (with 3D surface plot)
        Random Forest Regressor
        XGBoost Regressor
        Support Vector Regressor (SVR) with scaling and evaluation metrics:
        MSE, RMSE, R², MAE

🔹 Visualizations
        Age vs Charges (Linear Regression)
        3D surface plot of Age + BMI vs Charges (Decision Tree)
        SVR prediction vs actual scatter plot

1. Install Requirements
    pip install pandas numpy matplotlib scikit-learn xgboost

2. Download Dataset
    You can use kagglehub or download manually and place insurance.csv in your project directory.

3. Run Script
    python insurance_prediction.py
