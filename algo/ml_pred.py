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
from collections import Counter
import warnings
from sklearn.exceptions import DataConversionWarning

# Filter out UserWarnings from scikit-learn
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

# Filter out DataConversionWarnings from scikit-learn
warnings.filterwarnings(action='ignore', category=DataConversionWarning, module='sklearn')
 
# Reading the train.csv by removing the
# last column since it's an empty column
DATA_PATH = "algo/Training.csv"
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

import joblib

# Load the SVM model
final_svm_model = joblib.load('models/final_svm_model.pkl')

# Load the Naive Bayes model
final_nb_model = joblib.load('models/final_nb_model.pkl')

# Load the Random Forest model
final_rf_model = joblib.load('models/final_rf_model.pkl')
 

symptoms = X.columns.values
 
# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
 
data_dict = { #symptoms and diseases 
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}
 
# Defining the Function
# Input: string containing symptoms separated by commmas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
     
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    # reshaping the input data and converting it into suitable format for model predictions
    types_of_symptoms = input_data
    input_data = np.array(input_data).reshape(1,-1)
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    mode_pred = [rf_prediction, nb_prediction, svm_prediction]
    counter = Counter(mode_pred)
    top_k_modes = []
    unique_pred = set()
    for pred, count in counter.most_common(3):
      if len(top_k_modes) >= 3 or pred in unique_pred:
        continue
      top_k_modes.append(pred)
      unique_pred.add(pred)
    voted_pred = top_k_modes
    if len(voted_pred) == 1:
      voted_pred.append(voted_pred[0])
      voted_pred.append(voted_pred[0])
    if len(voted_pred) == 2:
      voted_pred.append(voted_pred[1])
    # making final prediction by taking mode of all predictions
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":voted_pred[0]
    }
    print("Predicted diagnosis ranking: \n" , "1. " , rf_prediction ,"\n", "2. " ,nb_prediction, "\n" , "3. " ,svm_prediction)
    print("List of Top Predictions: ", voted_pred, "Lenght of List:",  len(top_k_modes))
    return voted_pred