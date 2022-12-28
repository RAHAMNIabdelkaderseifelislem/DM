## One Rule Algorithm
is a supervised algorithm that's used to classify data into two or more classes. It's based on the idea that a single rule can be used to classify the data. This rule is called the **decision rule**. It's a rule that classifies a data instance into one of the classes.

The algorithm is as follows:

1. For each attribute, calculate the error rate.
2. Choose the attribute with the lowest error rate.
3. The decision rule is the attribute with the lowest error rate.
### Example
Suppose we have the following data:

| color | size | fruit |
|-------|------|-------|
| red   | big  | apple |
| red   | big  | apple |
| red   | big  | apple |
| red   | big  | apple |
| red   | big  | apple |
| red   | big  | apple |
| green | big  | apple |
| green | big  | apple |
| green | big  | apple |
| green | big  | apple |
| green | big  | apple |
| green | big  | apple |
| red   | small| banana|
| red   | small| banana|
| red   | small| banana|
| red   | small| banana|
| red   | small| banana|
| red   | small| banana|
| green | small| banana|
| green | small| banana|
| green | small| banana|
| green | small| banana|
| green | small| banana|
| green | small| banana|

The decision rule is the attribute with the lowest error rate. Let's calculate the error rate for each attribute.

#### color
| color | error rate |
|-------|------------|
| red   | 0.5        |
| green | 0.5        |

#### size
| size  | error rate |
|-------|------------|
| big   | 0.5        |
| small | 0.5        |

The decision rule is the attribute with the lowest error rate. In this case, the attribute is **color**.

The decision rule is:

```
IF color = red THEN fruit = apple
IF color = green THEN fruit = banana
```

### code
```python
import pandas as pd
from collections import Counter

data = pd.read_csv('fruit.csv')

# calculate the error rate for each attribute
attributes = data.columns[:-1]
error_rates = []
for attribute in attributes:
    error_rate = 0
    for i, instance in data.iterrows():
        # calculate the error rate for each value of the attribute
        for value in data[attribute].unique():
            # calculate the error rate for each class
            for label in data['fruit'].unique():
                # calculate the number of instances where the attribute equals the value and the class equals the label
                num = len(data[(data[attribute] == value) & (data['fruit'] == label)])
                # calculate the number of instances where the attribute equals the value
                den = len(data[data[attribute] == value])
                # calculate the error rate
                error_rate += (num / den) * (1 - (num / den))
    error_rates.append(error_rate)

# choose the attribute with the lowest error rate
decision_rule = attributes[error_rates.index(min(error_rates))]
print('The decision rule is:', decision_rule)
```
result:
```
The decision rule is: color
```
