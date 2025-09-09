import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load dataset
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# Dataset info
print("Dataset shape:", diabetes_dataset.shape)
print("\nOutcome counts:\n", diabetes_dataset['Outcome'].value_counts())
print("\nMean values per class:\n", diabetes_dataset.groupby('Outcome').mean())

# Split features & labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)   # FIXED axis issue
Y = diabetes_dataset['Outcome']

# Data standardization
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print("\nTrain/Test Split Shapes:")
print("X:", X.shape, "X_train:", X_train.shape, "X_test:", X_test.shape)

# Train SVM Classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("\nAccuracy score of the training data:", training_data_accuracy)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy score of the test data:", test_data_accuracy)

# Confusion Matrix & Report
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, X_test_prediction))
print("\nClassification Report:\n", classification_report(Y_test, X_test_prediction))

# Predictions
def predict_diabetes(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1,-1)
    std_data = scaler.transform(input_data_as_numpy_array)
    prediction = classifier.predict(std_data)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

# test input
sample_input = (10,101,86,37,0,45.6,1.136,38)
print("\nPrediction for sample input:", predict_diabetes(sample_input))

# Save Model
filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
print("\nModel saved as:", filename)
