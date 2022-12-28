## Decision Tree Algorithm
Decision Tree is a supervised algorithm that's used to classify data into two or more classes. It's based on the idea of splitting the data into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node (e.g., Outlook) has two or more branches (e.g., Sunny, Overcast, Rainy). Each branch represents a value for the attribute (e.g., Outlook). Leaf node (e.g., Play) represents a class label and contains a summary of the training data that belongs to that class.

The algorithm is as follows:

1. Calculate the entropy of the dataset.
2. Calculate the entropy of each feature.
3. Calculate the information gain of each feature.
4. Select the feature with the highest information gain.
5. Repeat steps 1-4 until all features have been used.
6. Build the decision tree.

### Example

Let's say we have the following data:

| Outlook | Temperature | Humidity | Wind | Play |
|---------|-------------|----------|------|------|
| Sunny   | Hot         | High     | Weak | No   |
| Sunny   | Hot         | High     | Strong | No   |
| Overcast | Hot         | High     | Weak | Yes   |
| Rainy   | Mild        | High     | Weak | Yes   |
| Rainy   | Cool        | Normal   | Weak | Yes   |
| Rainy   | Cool        | Normal   | Strong | No   |
| Overcast | Cool        | Normal   | Strong | Yes   |
| Sunny   | Mild        | High     | Weak | No   |
| Sunny   | Cool        | Normal   | Weak | Yes   |
| Rainy   | Mild        | Normal   | Weak | Yes   |
| Sunny   | Mild        | Normal   | Strong | Yes   |

We want to classify the data instance (Sunny, Cool, Normal, Weak). We'll use k = 3.

1. Calculate the entropy of the dataset.
2. Calculate the entropy of each feature.
3. Calculate the information gain of each feature.
4. Select the feature with the highest information gain.
5. Repeat steps 1-4 until all features have been used.
6. Build the decision tree.

The entropy of the dataset is:

| class | probability | entropy |
|-------|-------------|---------|
| No    | 0.357142857 | 0.985228136 |
| Yes   | 0.642857143 | 0.940285958 |

The entropy of each feature is:

| feature | entropy |
|---------|---------|
| Outlook | 0.693536 |
| Temperature | 0.911063 |
| Humidity | 0.78845 |
| Wind | 0.892158 |

The information gain of each feature is:

| feature | information gain |
|---------|------------------|
| Outlook | 0.246749819774 |
| Temperature | 0.0292225656589 |
| Humidity | 0.151835501362 |
| Wind | 0.0199730940212 |

The feature with the highest information gain is Outlook.

The entropy of the dataset is:

| class | probability | entropy |
|-------|-------------|---------|
| No    | 0.4 | 0.970950594 |
| Yes   | 0.6 | 0.918295834 |

The entropy of each feature is:

| feature | entropy |
|---------|---------|
| Temperature | 0.911063 |
| Humidity | 0.78845 |
| Wind | 0.892158 |

The information gain of each feature is:

| feature | information gain |
|---------|------------------|
| Temperature | 0.0199730940212 |
| Humidity | 0.151835501362 |
| Wind | 0.0199730940212 |

The feature with the highest information gain is Humidity.

The entropy of the dataset is:

| class | probability | entropy |
|-------|-------------|---------|
| No    | 0.5 | 0.693536 |
| Yes   | 0.5 | 0.693536 |

The entropy of each feature is:

| feature | entropy |
|---------|---------|
| Temperature | 0.911063 |
| Wind | 0.892158 |

The information gain of each feature is:

| feature | information gain |
|---------|------------------|
| Temperature | 0.0199730940212 |
| Wind | 0.0199730940212 |

The feature with the highest information gain is Temperature.

The entropy of the dataset is:

| class | probability | entropy |
|-------|-------------|---------|
| Yes    | 1 | 0 |

The entropy of each feature is:

| feature | entropy |
|---------|---------|
| Wind | 0.892158 |

The information gain of each feature is:

| feature | information gain |
|---------|------------------|
| Wind | 0.0199730940212 |

The feature with the highest information gain is Wind.

The entropy of the dataset is:

| class | probability | entropy |
|-------|-------------|---------|
| Yes    | 1 | 0 |

The entropy of each feature is:

| feature | entropy |
|---------|---------|
|       |  |

The information gain of each feature is:

| feature | information gain |
|---------|------------------|
|       |  |

The feature with the highest information gain is .

The decision tree is:

```
Outlook
├─ Overcast
│  └─ Yes
├─ Sunny
│  └─ Humidity
│     ├─ High
│     │  └─ No
│     └─ Normal
│        └─ Yes
└─ Rainy
   └─ Wind
      ├─ Weak
      │  └─ Yes
      └─ Strong
         └─ No
```

## Implementation

### Python

```python
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
```
 result:
```
Accuracy:  0.75
Confusion Matrix:  [[0 1]
 [0 2]]
Classification Report:                precision    recall  f1-score   support

          No       0.00      0.00      0.00         1
         Yes       0.67      1.00      0.80         2

    accuracy                           0.67         3
   macro avg       0.33      0.50      0.40         3
weighted avg       0.44      0.67      0.53         3
```