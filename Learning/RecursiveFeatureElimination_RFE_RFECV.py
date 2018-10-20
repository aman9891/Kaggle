import warnings
import pandas as pd
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split

# To supress the warnings in console output
warnings.filterwarnings("ignore")

# Path for files
training_file_path = "train.csv"
test_file_path = "test.csv"

# Reading training data from file
training_data = pd.read_csv(training_file_path)
features_training_data = training_data.iloc[:,0:10]
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

# RFE
final_training_data1 = final_training_data

# External estimator
estimator_rfe = SVR(kernel="linear")

# Recursive Feature Elimination
selector_rfe = RFE(estimator_rfe, n_features_to_select=5, step=2)
selector_rfe = selector_rfe.fit(final_training_data1, train_Y)
new_x_rfe = selector_rfe.transform(final_training_data1)

# Get the selected columns after RFE
mask_rfe = selector_rfe.get_support()
selected_columns_rfe = final_training_data1.columns[mask_rfe]

# Store the ranking of all the features and coefficients of the final selected features
result_ranking_rfe = pd.DataFrame(data=selector_rfe.ranking_, index=final_training_data1.columns)
result_coef_rfe = pd.DataFrame(data=selector_rfe.estimator_.coef_, columns=selected_columns_rfe)

print("The selected columns for RFE are :", selected_columns_rfe)

# RFECV
final_training_data2 = final_training_data

# External estimator
estimator_rfecv = SVR(kernel="linear")

# Recursive Feature Elimination
selector_rfecv = RFECV(estimator_rfecv, step=2, min_features_to_select=2, cv=5)
selector_rfecv = selector_rfecv.fit(final_training_data2, train_Y)
new_x_rfecv = selector_rfecv.transform(final_training_data2)

# Get the selected columns after RFE
mask_rfecv = selector_rfecv.get_support()
selected_columns_rfecv = final_training_data2.columns[mask_rfecv]

# Store the ranking of all the features and coefficients of the final selected features
result_ranking_rfecv = pd.DataFrame(data=selector_rfecv.ranking_, index=final_training_data2.columns)
result_coef_rfecv = pd.DataFrame(data=selector_rfecv.estimator_.coef_, columns=selected_columns_rfecv)

print("The selected columns for RFECV are :", selected_columns_rfecv)
print("Optimal number of features after REFCV are :", selector_rfecv.n_features_)