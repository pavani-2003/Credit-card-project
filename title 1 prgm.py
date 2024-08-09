import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the credit card fraud dataset (replace 'creditcard.csv' with your dataset)
data = pd.read_csv("C:\\Users\\pavani.k\\OneDrive\\Desktop\\project\\d1.csv")

# Data Preprocessing
# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in the dataset:\n", missing_values)

# Drop rows with any missing values
data.dropna(inplace=True)

# Check the data types of the columns
print("Data types of columns:\n", data.dtypes)

# Ensure that all columns are numeric
data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

# Split the data into features (X) and labels (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Standardize features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

# Predictions
knn_predictions = knn_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)

# Evaluation
print("K-Nearest Neighbors (KNN) Classifier Results:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_predictions))
print("Classification Report:\n", classification_report(y_test, knn_predictions))

print("\nSupport Vector Machine (SVM) Classifier Results:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))
print("Classification Report:\n", classification_report(y_test, svm_predictions))

# Visualization (you can customize this based on your needs)
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(confusion_matrix(y_test, knn_predictions), labels=['Non-Fraud', 'Fraud'])
plot_confusion_matrix(confusion_matrix(y_test, rf_predictions), labels=['Non-Fraud', 'Fraud'])
