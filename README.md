# Diabetes-Prediction-System
This project predicts diabetes using SVM (Support Vector Machine) on the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set?resource=download).
## Features
- Load and explore dataset (shape, class distribution, mean values per class)
- Standardize feature values using `StandardScaler`
- Split dataset into **train** and **test** sets (80/20) with stratification
- Train an **SVM classifier** (linear kernel)
- Evaluate the model using:
  - **Accuracy**
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-score)
- Make predictions for new inputs via a reusable function
- Save the trained model using `pickle`


## Results
- **Training Accuracy**: ~0.79  
- **Test Accuracy**: ~0.77  
- Confusion matrix and classification report show good performance for non-diabetic predictions, with room for improvement on diabetic recall.
