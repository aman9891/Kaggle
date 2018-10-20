import warnings
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

# To supress the warnings in console output
warnings.filterwarnings("ignore")

# Path for files
training_file_path = "train.csv"
test_file_path = "test.csv"

# Reading training data from file
training_data = pd.read_csv(training_file_path)
features_training_data = training_data.iloc[:,0:80]
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

# Univariate Feature Selection
# Using SelectKBest
final_training_data1 = final_training_data
selector_kbest = SelectKBest(mutual_info_regression, k = 20).fit(final_training_data1, train_Y)
new_x_kbest = selector_kbest.transform(final_training_data1)
scores_kbest = selector_kbest.scores_
pvalues_kbest = selector_kbest.pvalues_

# For naming the indices
scores_df_kbest = pd.DataFrame(data=scores_kbest, index=final_training_data1.columns)
pvalues_df_kbest = pd.DataFrame(data=pvalues_kbest, index=final_training_data1.columns)

# Getting the selected features
mask_kbest = selector_kbest.get_support()
new_features_kbest = final_training_data1.columns[mask_kbest]
print("Features selected by SelectKBest :", new_features_kbest)

# Using SelectPercentile
final_training_data2 = final_training_data
selector_perc = SelectPercentile(mutual_info_regression, percentile=7).fit(final_training_data2, train_Y)
new_x_perc = selector_perc.transform(final_training_data2)
scores_perc = selector_perc.scores_
pvalues_perc = selector_perc.pvalues_

# For naming the indices
scores_df_perc = pd.DataFrame(data=scores_perc, index=final_training_data2.columns)
pvalues_df_perc = pd.DataFrame(data=pvalues_perc, index=final_training_data2.columns)

# Getting the selected features
mask_perc = selector_perc.get_support()
new_features_perc = final_training_data.columns[mask_perc]
print("Features selected by SelectPercentile :", new_features_perc)