import warnings
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# To supress the warnings in console output
warnings.filterwarnings("ignore")

# Path for training file
training_file_path = "train.csv"

# --------------------------------Training Data--------------------------------------------------------

# Reading training data from file
training_data = pd.read_csv(training_file_path)

# Removing observations missing label value
training_data = training_data.dropna(axis = 0, subset = ["Survived"])

# Dividing data into features (we are ignoring Id column during training) and label
features_training_data = training_data.iloc[:,2:12]
label_training_data = training_data.Survived

# Splitting whole training data into training and vaidation data
train_X, val_X, train_Y, val_Y = train_test_split(features_training_data, label_training_data, train_size = 0.8, random_state = 1)

# Imputing of training data
train_X['Sex'] = (train_X['Sex'] == 'male').astype(int) # Imputing Sex Feature [Male = 1 and Female = 0]
train_X['Has_Cabin'] = (train_X['Cabin'].notnull()).astype(int) # Imputing Cabin Feature [Has Cabin = 1 and No Cabin = 0]
train_X['Total_Family_Members'] = (train_X['SibSp'] + train_X['Parch']) # Imputing number of siblings and number of parents
train_X['Age'].fillna(round(train_X['Age'].mean()), inplace=True) # Imputing Age and Fare with mean values of the column
train_X.drop(columns=['Cabin', 'SibSp', 'Parch', 'Name', 'Ticket', 'Embarked'], inplace=True) # Dropping unnecessary columns, which are already imputed

# Using RFE with RandomForest
"""estimator_rfe = RandomForestClassifier(random_state=1)
estimator_rfe.fit(train_X, train_Y)
print(estimator_rfe.feature_importances_)"""

# Training data with only selected features
# selected_features = ["Fare", "Sex", "Pclass", "Total_Family_Members"]
selected_features = ["Fare", "Total_Family_Members", "Sex", "Pclass"]
training_data_with_selected_features = train_X[selected_features]

# Training the model
random_forest_model = RandomForestClassifier(random_state=1)
random_forest_model.fit(training_data_with_selected_features, train_Y)

# -----------------------------------Validation Data---------------------------------------------------

# Imputing of Validation Data
val_X['Sex'] = (val_X['Sex'] == 'male').astype(int) # Imputing Sex Feature [Male = 1 and Female = 0]
val_X['Total_Family_Members'] = (val_X['SibSp'] + val_X['Parch']) # Imputing number of siblings and number of parents

# Validation data with selected features
validation_data_with_selected_features = val_X[selected_features]

# Predictions on Validation data
validation_data_predictions = random_forest_model.predict(validation_data_with_selected_features)
print("The error on validation data using Random Forest is =", mean_absolute_error(val_Y, validation_data_predictions))