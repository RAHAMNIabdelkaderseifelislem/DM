## Naive Bayes Algorithm

Naive Bayes is a supervised algorithm that's used to classify data into two or more classes. It's based on the idea of calculating the probability of a data instance belonging to a class.

The algorithm is as follows:

1. Calculate the probability of each class.
2. Calculate the probability of each feature given the class.
3. Calculate the probability of the data instance given the class.
4. The class of the data instance is the class with the highest probability.

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

1. Calculate the probability of each class.
2. Calculate the probability of each feature given the class.
3. Calculate the probability of the data instance given the class.
4. The class of the data instance is the class with the highest probability.

The probabilities are:

| class | probability |
|-------|-------------|
| 1     | 0.5         |
| 2     | 0.5         |

The probabilities of the features given the class are:

| class | x | y |
|-------|---|---|
| 1     | 1 | 1 |
| 2     | 1 | 1 |

The probability of the data instance given the class is:

| class | probability |
|-------|-------------|
| 1     | 0.5         |
| 2     | 0.5         |

The class of the data instance is the class with the highest probability, which is 1.

### Implementation

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['x', 'y']], data['class'], test_size=0.2, random_state=42)

# Create the model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)

print(accuracy)
```

### Output

```text
1.0
```
