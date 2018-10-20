import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# To supress the warnings in console output
warnings.filterwarnings("ignore")

# Path for files
training_file_path = "train.csv"
test_file_path = "test.csv"

# Reading training data from file
training_data = pd.read_csv(training_file_path)

# Removing observations missing label value
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

# Imputing numerical data
simple_numerical_imputer = SimpleImputer()
imputed_numerical_training_data = simple_numerical_imputer.fit_transform(numerical_training_data)
final_numerical_training_data = pd.DataFrame(imputed_numerical_training_data, columns = numerical_training_data.columns)

# One-hot encoding for categorical data
imputed_categorical_training_data = pd.get_dummies(categorical_training_data)

# This is very important to use before concat() function
final_numerical_training_data.reset_index(drop=True, inplace=True)
imputed_categorical_training_data.reset_index(drop=True, inplace=True)

# Combine imputed data back into single data
final_training_data = pd.concat([final_numerical_training_data, imputed_categorical_training_data], axis=1)

# Making pipeline for RandomForest and using Cross-Validation
random_forest_pipeline = make_pipeline(RandomForestRegressor(random_state = 1))
"""random_forest_pipeline.fit(final_training_data, train_Y)
scores = cross_val_score(random_forest_pipeline, val_X, scoring="neg_mean_absolute_error")

# Finding average score
avg_score = -1*scores.mean()
print("The error using Random Forest Regression with Cross Validation is =", avg_score)"""