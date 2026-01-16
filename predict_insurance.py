import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('insurance.csv')

categorical = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoder = encoder.fit_transform(df[categorical])
one_hot_df = pd.DataFrame(one_hot_encoder, columns=encoder.get_feature_names_out(categorical))
df_encoded = pd.concat([df, one_hot_df], axis=1)
insurance = df_encoded.drop(categorical, axis=1)

X = insurance.drop('charges', axis=1)
y = insurance[['charges']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 50 points between max and min values
age = np.linspace(insurance['age'].min(), insurance['age'].max(), 50)
bmi = np.linspace(insurance['bmi'].min(), insurance['bmi'].max(), 50)

# Makes two 2D arrays of possible pairings of age and bmi values
age_grid, bmi_grid = np.meshgrid(age, bmi)

# Defining fixed values for other features
fixed_values = {
    'children' : insurance['children'].median(),
    'sex_male'  : 1,
    'sex_female': 0,
    'smoker_no' : 1,
    'smoker_yes': 0,
    'northeast' : 0,
    'northwest' : 0,
    'southeast' : 1,
    'southwest' : 0
}

# Combining the grids into a single 2D array for prediction
grid = np.c_[
    age_grid.ravel(),
    bmi_grid.ravel(),
]

# Adding the fixed features as additional columns
for index, value in list(fixed_values.items()):
    fixed_array = np.full_like(age_grid.ravel(), fixed_values[index]).reshape(-1, 1)
    # Horizontally stack it onto grid
    grid = np.hstack([grid, fixed_array])


models = {
    "Random Forest" : RandomForestRegressor(),
    "XGB" : XGBRegressor(),
    "Decision Tree" : DecisionTreeRegressor(max_depth=5),
    "svr" : SVR(kernel='rbf', C=100, gamma='auto')
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Trying out different models and calculating the scores. Using 5 folds
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
        print(f"Model: {model_name}")
        print(f"Scores: {score}")
        print(f"Mean Accuracy: {score.mean():.4f}")
    except Exception as e:
        print("failure")

# Different scalers to avoid data leakage
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X_train)
y_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Making sure the right smoker and sex will be assigned
def data_clean_helper(sex_col, smoker_col):
    other_sex = "sex_female" if sex_col == "sex_male" else "sex_male"
    other_smoker = "smoker_no" if smoker_col == "smoker_yes" else "smoker_yes"
    
    return other_sex, other_smoker

# One region will have the value 1.0, the rest will have 0.0
def define_region(data, value_index, region_col):
    regions = ["northeast", "northwest", "southeast", "southwest"]

    for region in regions:
        if region == region_col:
            data[value_index[region_col]] = 1.0
        else:
            data[value_index["region_" + region]] = 0.0
    
    return data

values = X.columns.values
value_index = {col: i for i, col in enumerate(values)}

def data_clean(input):
    # Splitting the input values to handle the data
    input_values = input.split(",")
    data = [0] * len(value_index)

    # Depending on the column, inputs are assigned differently
    data[value_index['age']] = int(input_values[0])
    data[value_index['bmi']] = float(input_values[1])
    data[value_index['children']] = int(input_values[2])

    # Categorical values are taken and added to the rest of the column name
    sex_col = "sex_" + input_values[3].strip().lower()
    smoker_col = "smoker_" + input_values[4].strip().lower()
    region_col = "region_" + input_values[5].strip().lower()

    # Getting the sex and smoker columns
    other_sex, other_smoker = data_clean_helper(sex_col, smoker_col)

    # Assigning values for the sex and smoker columns
    data[value_index[sex_col]] = 1.0
    data[value_index[other_sex]] = 0.0

    data[value_index[smoker_col]] = 1.0
    data[value_index[other_smoker]] = 0.0

    # Assigning values for each region
    data = define_region(data, value_index, region_col)

    return data

# Function that predicts insurance using the data clean function
def predict_insurance(input):

    data = data_clean(input)

    scaled_data = X_scaler.fit_transform([data])
    
    # Make it back to the original scale to calculate metrics
    return svr.predict(scaled_data)[0]
