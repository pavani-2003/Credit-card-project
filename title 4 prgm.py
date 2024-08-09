import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the credit card fraud dataset (replace 'creditcard.csv' with your dataset)
data = pd.read_csv('C:\Users\pavani.k\OneDrive\Desktop\project\credit card.csv')

# Data Preprocessing
# Handle non-finite values (NaN or infinite)
data.dropna(inplace=True)
data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

# Split the data into features (X) and labels (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Logistic Regression Classifier
logistic_classifier = LogisticRegression(random_state=42)
logistic_classifier.fit(X_train, y_train)

# Predictions
knn_predictions = knn_classifier.predict(X_test)
logistic_predictions = logistic_classifier.predict(X_test)

# Evaluation
print("K-Nearest Neighbors (KNN) Classifier Results:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_predictions))
print("Classification Report:\n", classification_report(y_test, knn_predictions))

print("\nLogistic Regression Classifier Results:")
print("Accuracy:", accuracy_score(y_test, logistic_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, logistic_predictions))
print("Classification Report:\n", classification_report(y_test, logistic_predictions))

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
plot_confusion_matrix(confusion_matrix(y_test, logistic_predictions), labels=['Non-Fraud', 'Fraud'])
