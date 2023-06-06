from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("last_one.csv", skipinitialspace=True, low_memory=False)



def detect_outliers(df, column):
   
    threshold = 3
    outliers = []

    mean = np.mean(df[column])
    std = np.std(df[column])

    for value in df[column]:
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(str(value))

    return outliers
# Call the detect_outliers() function for the 'Price' column
outliers = detect_outliers(df, 'Price')

# Create a DataFrame from outliers
outliers_df = pd.DataFrame({'Outliers': outliers})

# Convert outliers to the same data type as 'Price' column
outliers = df['Price'].dtype.type(outliers)

# Filter the DataFrame to remove outliers
df_filtered = df[~df['Price'].isin(outliers)]

# Call the detect_outliers() function for the 'Living area' column
outliers_Bedrooms = detect_outliers(df, 'Bedrooms')

# Create a DataFrame from outliers
outliers_Bedrooms_df = pd.DataFrame({'outliers_Bedrooms': outliers_Bedrooms})

# Convert outliers to the same data type as 'Living area' column
outliers_Bedrooms = df['Bedrooms'].dtype.type(outliers_Bedrooms)

# Filter the DataFrame to remove outliers
df = df[~df['Bedrooms'].isin(outliers_Bedrooms)]

# Call the detect_outliers() function for the 'Living area' column
outliers_Primary_energy_consumption = detect_outliers(df, "Primary energy consumption")

# Create a DataFrame from outliers
outliers_Primary_energy_consumption_df = pd.DataFrame({'outliers_Primary_energy_consumption': outliers_Primary_energy_consumption })

# Convert outliers to the same data type as 'Living area' column
outliers_Primary_energy_consumption  = df["Primary energy consumption"].dtype.type(outliers_Primary_energy_consumption )

# Filter the DataFrame to remove outliers
df = df[~df['Bedrooms'].isin(outliers_Primary_energy_consumption)]

df['Apartments'] = (df['Subtype'].isin(['Apartment', 'Duplex'])).astype(int)

df = df.drop('Subtype', axis=1)


df = df[['Price', 'Bedrooms', 'Living area', 'Postcode', "Number of floors", "Floor", "Apartments", "Primary energy consumption", "Heating type", 'Construction year', 'Kitchen type', 'Region', 'Building condition']]


dummy_df = pd.get_dummies(df['Region']).astype(int)


df = df.drop('Region', axis=1)

df = pd.concat([df, dummy_df], axis=1)


dummy_df = pd.get_dummies(df['Building condition'], prefix='Building_condition').astype(int)

df = df.drop('Building condition', axis=1)

df = pd.concat([df, dummy_df], axis=1)

dummy_df = pd.get_dummies(df['Heating type'], prefix='Heating type').astype(int)

df = df.drop('Heating type', axis=1)

df = pd.concat([df, dummy_df], axis=1)


dummy_df = pd.get_dummies(df['Kitchen type'], prefix='Kitchen type').astype(int)

df = df.drop('Kitchen type', axis=1)

df = pd.concat([df, dummy_df], axis=1)

df = df.dropna()

#Preprocesseing , Feature Engeneering

X = df.drop("Price",axis=1)   #Feature Matrix
y = df["Price"] 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape

from sklearn.feature_selection import VarianceThreshold
var_thres=VarianceThreshold(threshold=0)
var_thres.fit(X_train)


constant_columns = [column for column in X_train.columns
                    if column not in X_train.columns[var_thres.get_support()]]

print(len(constant_columns))

for column in constant_columns:
    print(column)

X_train.drop(constant_columns,axis=1, inplace=True)

from sklearn.feature_selection import SelectPercentile


## Selecting the top 20 percentile
selected_top_columns = SelectPercentile(mutual_info_regression, percentile=20)
selected_top_columns.fit(X_train, y_train)

X_train.columns[selected_top_columns.get_support()]



df = df[[['Bedrooms', 'Living area', 'Postcode', 'Number of floors',
       'Primary energy consumption', 'Construction year']]]


#Prediction part 
# Trainning MODEL ML Engeneering

X = df.drop("Price",axis=1)   #Feature Matrix
y = df["Price"] 


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0)


import xgboost as xgb

reg  = xgb.XGBRegressor(base_score = 0.5,
                        booster = "gbtree",
                        objective = 'reg:squarederror',
                        max_depth = 3,
                        learning_rate = 0.05
                        )

reg.fit(X_train, y_train,
        eval_set = [(X_train, y_train)],
        verbose = 100
        )

reg.score(X_train, y_train)

reg.score(X_test, y_test)