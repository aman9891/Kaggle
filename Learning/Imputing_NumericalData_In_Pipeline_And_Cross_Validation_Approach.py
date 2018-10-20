import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# To supress the warnings in console output
warnings.filterwarnings("ignore")

# Path for files
training_file_path = "train.csv"
test_file_path = "test.csv"

# Reading training data from file
training_data = pd.read_csv(training_file_path)

# Removing observations missing label values
training_data = training_data.dropna(axis = 0, subset = ["SalePrice"])

# Dividing training data into features and label
features = ['LotArea', 'LotConfig', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
features_training_data = training_data[features]
label_training_data = training_data.SalePrice

# Splitting training data into training data and vaidation data
train_X, val_X, train_Y, val_Y = train_test_split(features_training_data, label_training_data, train_size = 0.8, random_state = 1)

# Diving data into numerical and categorical
numerical_training_data = train_X.select_dtypes(exclude=["object"])
categorical_training_data = train_X.select_dtypes(include=["object"])

# One-hot encoding for categorical data
imputed_categorical_training_data = pd.get_dummies(categorical_training_data)

# This is very important to use before concat() function
numerical_training_data.reset_index(drop=True, inplace=True)
imputed_categorical_training_data.reset_index(drop=True, inplace=True)

# Combine imputed data back into single data
modified_training_data = pd.concat([numerical_training_data, imputed_categorical_training_data], axis = 1)

# Made pipeline
# Using SimpleImputer for numerical data filling
# Getting neg_mean_absolute_error using Cross-Validation
random_forest_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state = 1))
"""random_forest_pipeline.fit(modified_training_data, train_Y)
scores = cross_val_score(random_forest_pipeline, modified_training_data, train_Y, scoring="neg_mean_absolute_error")

# Finding average score
avg_score = -1*scores.mean()
print("The error using Random Forest Regression with Cross Validation is =", avg_score)"""