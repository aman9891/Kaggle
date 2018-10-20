import warnings
import pandas as pd
from sklearn.impute import
from sklearn.ensemble import RandomForestRegressor

# To supress the warnings in console output
warnings.filterwarnings("ignore")

# Path for files
training_file_path = "train.csv"
test_file_path = "test.csv"

# --------------------------------Training Data--------------------------------------------------------

# Loading training data from file
training_data = pd.read_csv(training_file_path)

# Removing observations missing label value
training_data = training_data.dropna(axis = 0, subset = ["SalePrice"])

# Dividing data into features (we are ignoring Id column) and label (or actual output)
features_training_data = training_data.iloc[:,1:80]
label_training_data = training_data.SalePrice

# Diving training data into numerical and categorical
numerical_training_data = features_training_data.select_dtypes(exclude=["object"])
categorical_training_data = features_training_data.select_dtypes(include=["object"])

# Imputing numerical data
simple_training_numerical_imputer = SimpleImputer()
imputed_numerical_training_data = simple_training_numerical_imputer.fit_transform(numerical_training_data)
modified_numerical_training_data = pd.DataFrame(imputed_numerical_training_data, columns = numerical_training_data.columns)

# One-hot encoding for categorical data
modified_categorical_training_data = pd.get_dummies(categorical_training_data)

# This is very important to use before concat() function
modified_numerical_training_data.reset_index(drop=True, inplace=True)
modified_categorical_training_data.reset_index(drop=True, inplace=True)

# Combine imputed data back into single data
final_training_data = pd.concat([modified_numerical_training_data, modified_categorical_training_data], axis=1)

# Selecting training data with only selected features
selected_features = ["LotArea", "GrLivArea", "YearBuilt", "BsmtFinSF1", "OverallQual", "TotalBsmtSF", "1stFlrSF"]
training_data_with_selected_features = final_training_data[selected_features]

# Training the model (We already know RandomForest is giving best results)
random_forest_model = RandomForestRegressor(random_state=1)
random_forest_model.fit(training_data_with_selected_features, label_training_data)

# ------------------------------------Test Data---------------------------------------------------

# Reading test data from file
test_data = pd.read_csv(test_file_path)

# Diving test data into numerical and categorical
numerical_test_data = test_data.select_dtypes(exclude=["object"])
categorical_test_data = test_data.select_dtypes(include=["object"])

# Imputing of test numerical data
simple_test_numerical_imputer = SimpleImputer()
imputed_numerical_test_data = simple_test_numerical_imputer.fit_transform(numerical_test_data)
modified_numerical_test_data = pd.DataFrame(imputed_numerical_test_data, columns = numerical_test_data.columns)

# One-hot encoding for categorical test data
modified_categorical_test_data = pd.get_dummies(categorical_test_data)

# This is very important to use before concat() function
modified_numerical_test_data.reset_index(drop=True, inplace=True)
modified_categorical_test_data.reset_index(drop=True, inplace=True)

# Combine imputed data back into single test data
final_test_data = pd.concat([modified_numerical_test_data, modified_categorical_test_data], axis=1)

# Test data with selected features
test_data_with_selected_features = final_test_data[selected_features]

# Prediction on Test data
test_data_predictions = random_forest_model.predict(test_data_with_selected_features)

# Save your predictions to a file
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_data_predictions})
output.to_csv("submission.csv", index = False)
