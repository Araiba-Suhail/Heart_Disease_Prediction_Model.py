# Heart_Disease_Prediction_Model.py
Heart Disease Prediction Model

Project Overview
This project implements a machine learning model to predict the risk of heart disease based on clinical and demographic patient data. Using the UCI Heart Disease dataset, we build a logistic regression classifier to identify patients at risk, supporting early medical intervention.

Objective
Build a binary classification model to predict whether a person is at risk of heart disease based on health metrics, with comprehensive evaluation using accuracy, and confusion matrix.

Dataset
Source: UCI Heart Disease Dataset (Cleveland)
Samples: 303 patient records
Features: 13 clinical attributes + 1 target variable
Target: Presence of heart disease (binary: 0 = no disease, 1 = disease)

Feature Description:
- age: Age in years (Numerical)
- sex: Gender (1 = male; 0 = female) (Binary)
- cp: Chest pain type (1-4) (Categorical)
- trestbps: Resting blood pressure (mm Hg) (Numerical)
- chol: Serum cholesterol (mg/dl) (Numerical)
- fbs: Fasting blood sugar > 120 mg/dl (Binary)
- restecg: Resting electrocardiographic results (Categorical)
- thalach: Maximum heart rate achieved (Numerical)
- exang: Exercise induced angina (Binary)
- oldpeak: ST depression induced by exercise (Numerical)
- slope: Slope of peak exercise ST segment (Categorical)
- ca: Number of major vessels (0-3) (Categorical)
- thal: Thalassemia (3,6,7) (Categorical)
  
Installation & Setup
Prerequisites:
- Python 3.8+
- pip package manage
Install Dependencies:
-pip install pandas scikit-learn numpy matplotlib seaborn
Clone the repository:
-Clone repository git clone https://github.com/Araiba-Suahil/Heart_Disease_Prediction_Model.py.
-Navigate to directory cd Heart_Disease_Prediction_Model.py
Run the Model:
-python Heart_Disease_Prediction_Model.py


Implementation
Data Preprocessing:
1. Data Loading: Automatically fetches data from UCI repository
2. Missing Value Handling: Removes records with '?' values in 'ca' and 'thal' columns
3. Type Conversion: Converts 'ca' and 'thal' to float types
4. Target Variable: Transforms multi-class target (0-4) to binary (0 = no disease, 1 = disease)

Model Training:
- Algorithm: Logistic Regression
- Train-Test Split: 80-20 ratio with random state 42
- Features: All 13 clinical features
- Evaluation: Accuracy score and confusion matrix

Results
Model Performance:
The logistic regression model achieves accuracy (88.3 %) on the test set with a confusion matrix showing true/false positives and negatives.

Key Findings:
1. Dataset contains 303 records with 14 features
2. Missing values represented as '?' in 'ca' and 'thal' columns
3. Binary classification performed after data cleaning
4. Model trained and evaluated with standard metrics

Usage
Running the Full Pipeline:
python Heart_Disease_Prediction_Model.py

Making Predictions:
The script includes a prediction function that can be called with patient parameters:
- age: Patient age
- sex: Gender (1=male, 0=female)
- cp: Chest pain type (1-4)
- trestbps: Resting blood pressure
- chol: Cholesterol level
- fbs: Fasting blood sugar
- restecg: Resting ECG results
- thalach: Maximum heart rate
- exang: Exercise induced angina
- oldpeak: ST depression
- slope: Slope of ST segment
- ca: Number of major vessels
- thal: Thalassemia

Project Structure
The main script Heart_Disease_Prediction_Model.py contains:
1. Data loading and exploration
2. Data cleaning and preprocessing
3. Model training with logistic regression
4. Model evaluation with accuracy and confusion matrix
5. Prediction function for new patients

Code Features
- Automatic data fetching from UCI repository
- Comprehensive data exploration with statistics
- Missing value handling
- Binary target transformation
- Model training with scikit-learn
- Evaluation metrics calculation
- Ready-to-use prediction function

License
MIT License - See LICENSE file for details

Medical Disclaimer
This model is for educational and research purposes only. It should not be used for clinical diagnosis or medical decision-making. Always consult with healthcare professionals for medical advice.

Contributing:
-Fork the repository
-Create a feature branch
-Commit changes with descriptive messages
-Push to the branch
-Open a Pull Request
