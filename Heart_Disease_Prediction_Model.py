import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

column_names=[
    'age',      # Age in years
    'sex',      # Gender (1 = male; 0 = female)
    'cp',       # Chest pain type (1-4)
    'trestbps', # Resting blood pressure
    'chol',     # Serum cholesterol in mg/dl
    'fbs',      # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    'restecg',  # Resting electrocardiographic results
    'thalach',  # Maximum heart rate achieved
    'exang',    # Exercise induced angina (1 = yes; 0 = no)
    'oldpeak',  # ST depression induced by exercise
    'slope',    # Slope of the peak exercise ST segment
    'ca',       # Number of major vessels colored by fluoroscopy
    'thal',     # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
    'target'    # Presence of heart disease (0 = no; 1,2,3,4 = yes)
]

heart_data= pd.read_csv(url, names=column_names)
print("\n===Dataset BAsic Info===")
print(f"\nDataset shape:{heart_data.shape}")
print(f"\n==Column Names:{heart_data.columns}")
print("\n==First Five Rows of Datset==")
print(heart_data.head())
print("\n===Datatypes===")
print (heart_data.select_dtypes)
print("\n===Basic Statistics===")
print(heart_data.describe())
print ("\n===Dataset understaning completed===")

print("\n===Finidng Missing Values===")
print("\nNull Values Count")
print (f'\nThe number of Null values in this dataset is {heart_data.isnull().sum()}')

print (f'\nChecking for "?" as a missing data symbol')
for column in heart_data:
    have_mark = (heart_data[column] == "?").any()
    print(f'The column {column} has "?"') 
    mark_value_count= (heart_data[column] == "?").sum()
    print (f'The number of "?" in {column} is {mark_value_count}')

print ("\nTarget Variable Distribution")
print (f'This {heart_data["target"].value_counts()}')
print("\nValue = o means no disease, Value = 1-4 means heart disease.")

heart_data['heart_disease']=(heart_data['target'] >0).astype(int)
print('\n===Siplified Target Variable===')
print(heart_data['heart_disease'].value_counts())
print ("\n Value = 0 >> Heart Disease, Value = 1 >>> No Heart Disease")

print ("\nCleaning Data")
print("\n REmoving Rows with missing values")
heart_clean= heart_data(heart_data['ca'] != '?' & heart_data['thal'] != "?")
print (f' Shape of Original Dataset {heart_data.shape}.')
print (f' SHape of Cleaned Dataset {heart_clean}.')

#converting columns datatypes
heart_clean['ca']= heart_clean['ca'].astype(float)
heart_clean['thal']= heart_clean['thal'].astype(float)
print (heart_clean.dtypes )

#Preparing for machine learning
print('\n Preparing Data for machine Learning')
features= [
    'age',      # Age in years
    'sex',      # Gender (1 = male; 0 = female)
    'cp',       # Chest pain type (1-4)
    'trestbps', # Resting blood pressure
    'chol',     # Serum cholesterol in mg/dl
    'fbs',      # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    'restecg',  # Resting electrocardiographic results
    'thalach',  # Maximum heart rate achieved
    'exang',    # Exercise induced angina (1 = yes; 0 = no)
    'oldpeak',  # ST depression induced by exercise
    'slope',    # Slope of the peak exercise ST segment
    'ca',       # Number of major vessels colored by fluoroscopy
    'thal',     # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
    'target'    # Presence of heart disease (0 = no; 1,2,3,4 = yes)
]
X= heart_clean[features]
y= heart_clean["heart_disease"]

print ('\n Feartures (X) we are goign to use to predict')
print (f'\n Shape of X column is  {X.shape}.')
print (f'\n Column names are', list(X.columns))

print (f'\n Target (Y) we are going to predict')
print (f'\nShape', y.shape)
print (f'\n Value count')
print (y.value_counts())


#Model Training
print ("Training Logistic Regression Model")
X_train, X_test, y_train, y_test= train_test_split( X,y , random_state= 42, test_size= 0.2)
model= LogisticRegression (random_state= 42)
model.fit (X_train, y_train)
print (f'\n Training Completed ✔')
print (f'\n Making Predictions')
y_pred= model.predict(X_test)
print (f' Prediction Completed✔✔')
print (f'\nFirst 10 prediction{(y_test)[:10]}')
print (f'\nFIrst 10 actual values {(y_pred)[10:]}')

#Evaluation
print (f'\n Evaluating Model')
accuracy= accuracy_score (y_test, y_pred)
print (f'\n Accracy socre is {accuracy:.0f}, ({(accuracy)*100:.0f})')

cm = confusion_matrix ( y_test, y_pred)
print(cm)

#Predictions

def patient_heart_disease (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    patient_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict[ [patient_data][0]]
    probablity= model.predict.proba[patient_data][0][1]
    return "Heart_disease" if prediction == 0 else "No Heart Disease" f'{probablity:.2 %}'

#example usage
result, prob = predict_heart_disease(63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1)
print (f'Prediction: {result}, Probablity: {prob}')

