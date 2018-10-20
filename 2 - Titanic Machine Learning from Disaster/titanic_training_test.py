import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# To supress the warnings in console output
warnings.filterwarnings("ignore")

# Path for training file
training_file_path = "train.csv"
test_file_path = "test.csv"

# --------------------------------Training Data--------------------------------------------------------

# Reading training data from file
training_data = pd.read_csv(training_file_path)

# Removing observations missing label value
training_data = training_data.dropna(axis = 0, subset = ["Survived"])

# Dividing data into features (we are ignoring Id column during training) and label
features_training_data = training_data.iloc[:,2:12]
label_training_data = training_data.Survived

# Imputing of training data
features_training_data['Sex'] = (features_training_data['Sex'] == 'male').astype(int) # Imputing Sex Feature [Male = 1 and Female = 0]

features_training_data['Has_Cabin'] = (features_training_data['Cabin'].notnull()).astype(int) # Imputing Cabin Feature [Has Cabin = 1 and No Cabin = 0]

features_training_data['Total_Family_Members'] = (features_training_data['SibSp'] + features_training_data['Parch']) # Imputing number of siblings and number of parents

features_training_data['Age'].fillna(round(features_training_data['Age'].mean()), inplace=True) # Imputing Age with mean value of the column

features_training_data['Embarked'].fillna('C', inplace=True)
features_training_data['Port'] = features_training_data['Embarked'].map({'C':0, 'Q':1 , 'S':2}).astype(int)

features_training_data['Fare'].fillna(features_training_data['Fare'].mean(), inplace=True) # Imputing Fare with mean value of the column
features_training_data['Fare_Per_Person'] = np.where(features_training_data['Total_Family_Members']>0, features_training_data['Fare']/features_training_data['Total_Family_Members'], features_training_data['Fare'])

features_training_data.drop(columns=['Cabin', 'SibSp', 'Parch', 'Name', 'Ticket', 'Embarked', 'Fare'], inplace=True) # Dropping unnecessary columns, which are already imputed

# Training data with only selected features
selected_features = ["Fare_Per_Person", "Sex", "Total_Family_Members", "Has_Cabin", "Pclass", "Port"]
training_data_with_selected_features = features_training_data[selected_features]

# Training the model
random_forest_model = RandomForestClassifier(random_state=1)
random_forest_model.fit(training_data_with_selected_features, label_training_data)

# ------------------------------------Test Data---------------------------------------------------

# Reading test data from file
test_data = pd.read_csv(test_file_path)

# Imputing of test data
test_data['Sex'] = (test_data['Sex'] == 'male').astype(int) # Imputing Sex Feature [Male = 1 and Female = 0]

test_data['Has_Cabin'] = (test_data['Cabin'].notnull()).astype(int) # Imputing Cabin Feature [Has Cabin = 1 and No Cabin = 0]

test_data['Total_Family_Members'] = (test_data['SibSp'] + test_data['Parch']) # Imputing number of siblings and number of parents

test_data['Age'].fillna(round(test_data['Age'].mean()), inplace=True) # Imputing Age with mean value of the column

test_data['Embarked'].fillna('C', inplace=True)
test_data['Port'] = test_data['Embarked'].map({'C':0, 'Q':1 , 'S':2}).astype(int)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True) # Imputing Fare with mean value of the column
test_data['Fare_Per_Person'] = np.where(test_data['Total_Family_Members']>0, test_data['Fare']/test_data['Total_Family_Members'], test_data['Fare'])

test_data.drop(columns=['Cabin', 'SibSp', 'Parch', 'Name', 'Ticket', 'Embarked', 'Fare'], inplace=True) # Dropping unnecessary columns, which are already imputed

# Test data with selected features
test_data_with_selected_features = test_data[selected_features]

# Prediction on Test data
test_data_predictions = random_forest_model.predict(test_data_with_selected_features)

# Save your predictions to a file
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_data_predictions})
output.to_csv("submission.csv", index = False)