import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the credit card fraud dataset (replace 'credit_card_dataset.csv' with your dataset)
data = pd.read_csv("C:\\Users\\pavani.k\\OneDrive\\Desktop\\project\\d1.csv")

# Data Preprocessing
# Check for missing values
print("Missing values in the dataset:\n", data.isnull().sum())

# Split the data into features (X) and labels (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values in the target variable with SimpleImputer
y_train_imputer = SimpleImputer(strategy='most_frequent')
y_train = y_train_imputer.fit_transform(y_train.values.reshape(-1, 1))

# Handle missing values with SimpleImputer in the features
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Check for and handle infinite values
X_train = np.where(np.isfinite(X_train), X_train, np.nan_to_num(X_train))
X_test = np.where(np.isfinite(X_test), X_test, np.nan_to_num(X_test))

# Standardize features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Naive Bayes Classifier (GaussianNB)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predictions
knn_predictions = knn_classifier.predict(X_test)
nb_predictions = nb_classifier.predict(X_test)

# Evaluation
print("K-Nearest Neighbors (KNN) Classifier Results:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_predictions))
print("Classification Report:\n", classification_report(y_test, knn_predictions))

print("\nNaive Bayes Classifier (GaussianNB) Results:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_predictions))
print("Classification Report:\n", classification_report(y_test, nb_predictions))

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
plot_confusion_matrix(confusion_matrix(y_test, nb_predictions), labels=['Non-Fraud', 'Fraud'])
