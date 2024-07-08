# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
 
# Reading the train.csv by removing the
# last column since it's an empty column
DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)
 
# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

raw_diseases = pd.DataFrame(data['prognosis']) #get the list of all the diseases before they are encoded

# Encoding the target value into numerical
# value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Removing GERD
index_G = raw_diseases.loc[raw_diseases['prognosis'] == 'GERD'].index
for i in range(len(index_G)):
  data = data.drop(index=index_G[i])

# Removing AIDS
index = raw_diseases.loc[raw_diseases['prognosis'] == 'AIDS'].index
for i in range(len(index_G)):
  data = data.drop(index=index[i])

# Removing Hepatitus A, B, C, D, E
index_A = raw_diseases.loc[raw_diseases['prognosis'] == 'hepatitis A'].index
index_B = raw_diseases.loc[raw_diseases['prognosis'] == 'Hepatitis B'].index
index_C = raw_diseases.loc[raw_diseases['prognosis'] == 'Hepatitis C'].index
index_D = raw_diseases.loc[raw_diseases['prognosis'] == 'Hepatitis D'].index
index_E = raw_diseases.loc[raw_diseases['prognosis'] == 'Hepatitis E'].index

for i in range(len(index_A)):
  data = data.drop(index=index_A[i])

for i in range(len(index_B)):
  data = data.drop(index=index_B[i])

for i in range(len(index_C)):
  data = data.drop(index=index_C[i])

for i in range(len(index_D)):
  data = data.drop(index=index_D[i])

for i in range(len(index_E)):
  data = data.drop(index=index_E[i])

# Removing Chronic cholestasis, Alcoholic hepatitis, and Heart attack
index_C = raw_diseases.loc[raw_diseases['prognosis'] == 'Chronic cholestasis'].index
index_AH = raw_diseases.loc[raw_diseases['prognosis'] == 'Alcoholic hepatitis'].index
index_H = raw_diseases.loc[raw_diseases['prognosis'] == 'Heart attack'].index

for i in range(len(index_C)):
  data = data.drop(index=index_C[i])

for i in range(len(index_AH)):
  data = data.drop(index=index_AH[i])

for i in range(len(index_H)):
  data = data.drop(index=index_H[i])

X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
  X, y, test_size = 0.2, random_state = 24)
 
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# Defining scoring metric for k-fold cross validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))
 
# Initializing Models
models = {
    "SVC":SVC(),
    "Gaussian NB":GaussianNB(),
    "Random Forest":RandomForestClassifier(random_state=18)
}
 
# Producing cross validation score for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv = 10, n_jobs = -1,scoring = cv_scoring)
    print("=="*30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")

# Training and testing SVM Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)
 
print(f"Accuracy on train data by SVM Classifier\
: {accuracy_score(y_train, svm_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by SVM Classifier\
: {accuracy_score(y_test, preds)*100}")

print("=="*30)

# Training and testing Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)
print(f"Accuracy on train data by Naive Bayes Classifier\
: {accuracy_score(y_train, nb_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by Naive Bayes Classifier\
: {accuracy_score(y_test, preds)*100}")
 
print("=="*30)

# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
print(f"Accuracy on train data by Random Forest Classifier\
: {accuracy_score(y_train, rf_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by Random Forest Classifier\
: {accuracy_score(y_test, preds)*100}")
 
 # Reading the test data
test_data = pd.read_csv("Testing.csv").dropna(axis=1)

# Removing extreme diseases (based on assumptions from existing literature)
test_data = test_data[test_data['prognosis'] != 'GERD']
test_data = test_data[test_data['prognosis'] != 'hepatitis A']
test_data = test_data[test_data['prognosis'] != 'Hepatitis B']
test_data = test_data[test_data['prognosis'] != 'Hepatitis C']
test_data = test_data[test_data['prognosis'] != 'Hepatitis D']
test_data = test_data[test_data['prognosis'] != 'Hepatitis E']
test_data = test_data[test_data['prognosis'] != 'Alcoholic hepatitis']
test_data = test_data[test_data['prognosis'] != 'Heart attack']
test_data = test_data[test_data['prognosis'] != 'Chronic cholestasis']
test_data = test_data[test_data['prognosis'] != 'AIDS']


# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
 
# Making prediction by take mode of predictions
# made by all the classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)
 
final_preds = [mode([i, j, k], axis=None, keepdims=True)[0][0] for i, j, k in zip(svm_preds, nb_preds, rf_preds)]
 
import joblib

# Save the SVM model
joblib.dump(final_svm_model, '../models/final_svm_model.pkl')

# Save the Naive Bayes model
joblib.dump(final_nb_model, '../models/final_nb_model.pkl')

# Save the Random Forest model
joblib.dump(final_rf_model, '../models/final_rf_model.pkl')
