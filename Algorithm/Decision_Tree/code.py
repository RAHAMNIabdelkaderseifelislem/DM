import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the data
data = pd.read_csv('data.csv')

# Split the data into features and target
X = data.drop('Play', axis=1)
y = data['Play']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the decision tree
clf = tree.DecisionTreeClassifier(criterion='entropy')

# Train the decision tree
clf = clf.fit(X_train, y_train)

# Predict the target
y_pred = clf.predict(X_test)

# Calculate the accuracy
print('Accuracy: ', accuracy_score(y_test, y_pred))

# Calculate the confusion matrix
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))

# Calculate the classification report
print('Classification Report: ', classification_report(y_test, y_pred))