import pandas as pd
import warnings
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

# To supress the warnings in console output
warnings.filterwarnings("ignore")

# Path for files
training_file_path = "train.csv"
test_file_path = "test.csv"

# Reading training data from file
training_data = pd.read_csv(training_file_path)

# Find columns with missing data
col_with_missing_data = [col for col in training_data.columns if training_data[col].isnull().any() == True]
print("Following features have missing data :", col_with_missing_data)

# Dividing training data into features and label
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
features_training_data = training_data[features]
label_training_data = training_data.SalePrice

# Splitting training data into training data and vaidation data
train_X, val_X, train_Y, val_Y = train_test_split(features_training_data, label_training_data, train_size = 0.8, random_state = 1)

# Decision Tree Regression
decision_tree_pipeline = make_pipeline(DecisionTreeRegressor(random_state = 1))
decision_tree_pipeline.fit(train_X, train_Y)
decision_tree_pipeline_val_predictions = decision_tree_pipeline.predict(val_X)
print("The error on validation data using Decision Tree Regression is =", mean_absolute_error(val_Y, decision_tree_pipeline_val_predictions))

# Random Forest Regression
random_forest_pipeline = make_pipeline(RandomForestRegressor(random_state = 1))
random_forest_pipeline.fit(train_X, train_Y)
random_forest_pipeline_val_predictions = random_forest_pipeline.predict(val_X)
print("The error on validation data using Random Forest Regression is =", mean_absolute_error(val_Y, random_forest_pipeline_val_predictions))

# XGBoost Regression
xgboost_pipeline = make_pipeline(XGBRegressor())
xgboost_pipeline.fit(train_X, train_Y)
xgboost_pipeline_val_predictions = xgboost_pipeline.predict(val_X)
print("The error on validation data using XGBoost Regression is =", mean_absolute_error(val_Y, xgboost_pipeline_val_predictions))

# XGBoost Regression with Parameters
xgboost_parameters_pipeline = make_pipeline(XGBRegressor(n_estimators = 150))
xgboost_parameters_pipeline.fit(train_X, train_Y)
xgboost_parameters_pipeline_val_predictions = xgboost_parameters_pipeline.predict(val_X)
print("The error on validation data using XGBoost Regression with parameters is =", mean_absolute_error(val_Y, xgboost_parameters_pipeline_val_predictions))

# Reading test data and dividing into features and label
test_data = pd.read_csv("test.csv")
features_test_data = test_data[features]

# Fitting model on complete training data and calculating prediction on real test data
random_forest_model_all = RandomForestRegressor(random_state=1)
random_forest_model_all.fit(features_training_data, label_training_data)
test_predictions = random_forest_model_all.predict(features_test_data)

# Save your predictions to a file
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})
output.to_csv("submission.csv", index = False)