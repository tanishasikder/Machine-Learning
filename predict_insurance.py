import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Load the data
df = pd.read_csv('insurance.csv')

# Find the categorical data and one-hot encode it
categorical = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoder = encoder.fit_transform(df[categorical])
one_hot_df = pd.DataFrame(one_hot_encoder, columns=encoder.get_feature_names_out(categorical))
df_encoded = pd.concat([df, one_hot_df], axis=1)
insurance = df_encoded.drop(categorical, axis=1)

# Drop charges, use it to predict
X = insurance.drop('charges', axis=1)
y = insurance['charges']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# All the models that will be used
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "SVR": SVR(kernel='rbf', C=100, gamma='auto')
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 60)
print("Model Comparison (5-Fold Cross-Validation)")
print("=" * 60)

# Loop through all the models and use cross validation as a score
for model_name, model in models.items():
    try:
        score = cross_val_score(
            model,
            X, 
            y, 
            cv=kfold,
            scoring='r2',
            n_jobs=-1,
            error_score='raise'
        )
        print(f"\nModel: {model_name}")
        print(f"Scores: {score}")
        print(f"Mean R² Score: {score.mean():.4f} (+/- {score.std():.4f})")
    except Exception as e:
        print(f"\nModel: {model_name}")
        print(f"Error: {e}")

print("\n" + "=" * 60)

# Scaling 
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X_train)
y_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Training SVR model
svr = SVR(kernel='rbf', C=100, gamma='auto')
svr.fit(X_scaled, y_scaled)

# Also train on test set for evaluation
X_test_scaled = X_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

print("\nFinal SVR Model Training Complete")
print("=" * 60)

# Helper functions
def data_clean_helper(sex_col, smoker_col):
    # Determines the sex based on the user input
    other_sex = "sex_female" if sex_col == "sex_male" else "sex_male"
    # Determines smoking status based on suer input
    other_smoker = "smoker_no" if smoker_col == "smoker_yes" else "smoker_yes"
    return other_sex, other_smoker

def define_region(data, value_index, region_col):
    regions = ["region_northeast", "region_northwest", "region_southeast", "region_southwest"]
    
    # If the parameter region matches a region, it is marked with 1, otherwise 0
    for region in regions:
        if region == region_col:
            data[value_index[region]] = 1.0
        else:
            data[value_index[region]] = 0.0
    
    return data

# Create column index mapping
values = X.columns.values
value_index = {col: i for i, col in enumerate(values)}

def data_clean(input_str):
    input_values = input_str.split(",")
    data = [0] * len(value_index)

    # Numerical features
    data[value_index['age']] = float(input_values[0])
    data[value_index['bmi']] = float(input_values[1])
    data[value_index['children']] = int(input_values[2])

    # Categorical features 
    sex_col = "sex_" + input_values[3].strip().lower()
    smoker_col = "smoker_" + input_values[4].strip().lower()
    region_col = "region_" + input_values[5].strip().lower()

    # Set sex and smoker columns
    other_sex, other_smoker = data_clean_helper(sex_col, smoker_col)
    
    data[value_index[sex_col]] = 1.0
    data[value_index[other_sex]] = 0.0
    
    data[value_index[smoker_col]] = 1.0
    data[value_index[other_smoker]] = 0.0

    # Set region columns
    data = define_region(data, value_index, region_col)

    return data

def predict_insurance(input_str):
    data = data_clean(input_str)
    
    scaled_data = X_scaler.transform([data])
    
    # Get prediction in scaled space
    scaled_prediction = svr.predict(scaled_data)[0]
    
    # Inverse transform to get actual dollar amount
    actual_prediction = y_scaler.inverse_transform([[scaled_prediction]])[0][0]
    
    return actual_prediction