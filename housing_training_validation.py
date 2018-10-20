import warnings
import pandas as pd
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# To supress the warnings in console output
warnings.filterwarnings("ignore")

# Path for training file
training_file_path = "train.csv"

# --------------------------------Training Data--------------------------------------------------------

# Load training data from file
training_data = pd.read_csv(training_file_path)

# Removing observations missing label value
training_data = training_data.dropna(axis = 0, subset = ["SalePrice"])

# Dividing training data into features (we are ignoring Id column) and label (or actual output)
features_training_data = training_data.iloc[:,1:80]
label_training_data = training_data.SalePrice

# Splitting whole training data into training and vaidation data
train_X, val_X, train_Y, val_Y = train_test_split(features_training_data, label_training_data, train_size = 0.8, random_state = 1)

# Diving training data into numerical and categorical
numerical_training_data = train_X.select_dtypes(exclude=["object"])
categorical_training_data = train_X.select_dtypes(include=["object"])

# Imputing numerical data
simple_training_numerical_imputer = SimpleImputer()
imputed_numerical_training_data = simple_training_numerical_imputer.fit_transform(numerical_training_data)
modified_numerical_training_data = pd.DataFrame(imputed_numerical_training_data, columns = numerical_training_data.columns)

# One-hot encoding for categorical data
modified_categorical_training_data = pd.get_dummies(categorical_training_data)

# This is very important to use before concat() function
modified_numerical_training_data.reset_index(drop=True, inplace=True)
modified_categorical_training_data.reset_index(drop=True, inplace=True)

# Combine imputed data back into single dataframe
final_training_data = pd.concat([modified_numerical_training_data, modified_categorical_training_data], axis=1)

# Removing features with very low variance
variance_fs = VarianceThreshold(threshold = 0.1)
training_data_after_variance_fs = variance_fs.fit_transform(final_training_data)
training_data_after_variance_fs = pd.DataFrame(data=training_data_after_variance_fs, columns=final_training_data.columns[variance_fs.get_support()])

# Feature Selection by different models
# Using SelectKBest
"""selectkbest_fs = SelectKBest(mutual_info_regression, k = 10)
selectkbest_fs.fit_transform(training_data_after_variance_fs, train_Y)
print(training_data_after_variance_fs.columns[selectkbest_fs.get_support()])
print(selectkbest_fs.scores_[selectkbest_fs.get_support()])"""

# Using RFE with XGBoost
"""estimator_rfe = XGBRegressor(random_state=1)
rfe_fs = RFE(estimator_rfe, n_features_to_select=10, step=2)
rfe_fs.fit_transform(training_data_after_variance_fs, train_Y)
print(training_data_after_variance_fs.columns[rfe_fs.get_support()])
print(rfe_fs.estimator_.feature_importances_)"""

# Using RFE with RandomForest (as this model gives best result out of the three)
estimator_rfe = RandomForestRegressor(random_state=1)
rfe_fs = RFE(estimator_rfe, n_features_to_select=10, step=2)
rfe_fs.fit_transform(training_data_after_variance_fs, train_Y)
print(training_data_after_variance_fs.columns[rfe_fs.get_support()])
print(rfe_fs.estimator_.feature_importances_)

# Training data with only selected features
selected_features = ["LotArea", "GrLivArea", "YearBuilt", "BsmtFinSF1", "OverallQual", "TotalBsmtSF", "1stFlrSF"]
training_data_with_selected_features = training_data_after_variance_fs[selected_features]

# Training the model
random_forest_model = RandomForestRegressor(random_state=1)
random_forest_model.fit(training_data_with_selected_features, train_Y)

# -----------------------------------Validation Data---------------------------------------------------

# Dividing validation data into numerical and categorical
numerical_validation_data = val_X.select_dtypes(exclude=["object"])
categorical_validation_data = val_X.select_dtypes(include=["object"])

# Imputing of Validation data
simple_validation_numerical_imputer = SimpleImputer()
imputed_numerical_validation_data = simple_validation_numerical_imputer.fit_transform(numerical_validation_data)
modified_numerical_validation_data = pd.DataFrame(imputed_numerical_validation_data, columns = numerical_validation_data.columns)

# One-hot encoding for categorical validation data
modified_categorical_validation_data = pd.get_dummies(categorical_validation_data)

# This is very important to use before concat() function
modified_numerical_validation_data.reset_index(drop=True, inplace=True)
modified_categorical_validation_data.reset_index(drop=True, inplace=True)

# Combine imputed data back into single validation data
final_validation_data = pd.concat([modified_numerical_validation_data, modified_categorical_validation_data], axis=1)

# Validation data with selected features
validation_data_with_selected_features = final_validation_data[selected_features]

# Predictions on Validation data and calculating error rate  (it should be less)
validation_data_predictions = random_forest_model.predict(validation_data_with_selected_features)
print("The error on validation data using Random Forest is =", mean_absolute_error(val_Y, validation_data_predictions))