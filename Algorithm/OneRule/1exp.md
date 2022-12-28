## One Rule Algorithm
is a supervised algorithm that's used to classify data into two or more classes. It's based on the idea that a single rule can be used to classify the data. This rule is called the **decision rule**. It's a rule that classifies a data instance into one of the classes.

The algorithm is as follows:

1. For each attribute, calculate the entropy of the data.
2. Choose the attribute with the lowest entropy.
3. Use this attribute to create the decision rule.
4. Repeat steps 1-3 for each possible value of the chosen attribute.

### Example
Let's assume we have the following data:
+-------------+----------+----------+------+
| Temperature | Humidity | Wind     | Play |
+-------------+----------+----------+------+
|   Hot       |   High   | Weak     | No   |
|   Hot       |   High   | Strong   | No   |
|  Overcast   |   High   | Weak     | Yes  |
|   Rain      |  Normal  | Weak     | Yes  |
|   Rain      |    Low   | Weak     | Yes  |
|   Rain      |    Low   | Strong   | No   |
|  Overcast   |    Low   | Strong   | Yes  |
|   Hot       |  Normal  | Weak     | No   |
|   Hot       |    Low   | Weak     | Yes  |
|   Rain      |  Normal  | Weak     | Yes  |
|   Hot       |  Normal  | Strong   | Yes  |
|  Overcast   |  Normal  | Strong   | Yes  |
|  Overcast   |   High   | Weak     | Yes  |
|   Rain      |  Normal  | Strong   | No   |
+-------------+----------+----------+------+

