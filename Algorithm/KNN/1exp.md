## KNN Algorithm
is a supervised algorithm that's used to classify data into two or more classes. It's based on the idea of finding the nearest neighbors of a data instance and classifying it based on the majority of the neighbors' classes.

The algorithm is as follows:

1. Calculate the distance between the data instance and every other data instance.
2. Sort the distances in ascending order.
3. Choose the first k data instances.
4. The class of the data instance is the majority of the k data instances.

### Example
Let's say we have the following data:

| x | y | class |
|---|---|-------|
| 1 | 1 | 1     |
| 2 | 2 | 1     |
| 3 | 3 | 1     |
| 4 | 4 | 2     |
| 5 | 5 | 2     |
| 6 | 6 | 2     |

We want to classify the data instance (3, 3). We'll use k = 3.

1. Calculate the distance between the data instance and every other data instance.
2. Sort the distances in ascending order.
3. Choose the first k data instances.
4. The class of the data instance is the majority of the k data instances.

The distances are:

| x | y | class | distance |
|---|---|-------|----------|
| 1 | 1 | 1     | 2.828    |
| 2 | 2 | 1     | 2.828    |
| 3 | 3 | 1     | 0        |
| 4 | 4 | 2     | 2.828    |
| 5 | 5 | 2     | 2.828    |
| 6 | 6 | 2     | 5.657    |

The first three data instances are the closest to the data instance (3, 3). The class of the data instance is the majority of the k data instances, which is 1.

### Implementation
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['x', 'y']], data['class'], test_size=0.2, random_state=42)

# Create the model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)

print(accuracy)
```

### Output
```python
1.0
```
