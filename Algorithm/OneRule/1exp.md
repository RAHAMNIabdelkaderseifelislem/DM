## One Rule Algorithm
is a supervised algorithm that's used to classify data into two or more classes. It's based on the idea that a single rule can be used to classify the data. This rule is called the **decision rule**. It's a rule that classifies a data instance into one of the classes.

The algorithm is as follows:

1. For each attribute, calculate the entropy of the data.
2. Choose the attribute with the lowest entropy.
3. Use this attribute to create the decision rule.
4. Repeat steps 1-3 for each possible value of the chosen attribute.

### Example
Let's assume we have the following data:
+-------------+----------+----------+------+<br>
| Temperature | Humidity | Wind     | Play |<br>
+-------------+----------+----------+------+<br>
|   Hot       |   High   | Weak     | No   |<br>
|   Hot       |   High   | Strong   | No   |<br>
|  Overcast   |   High   | Weak     | Yes  |<br>
|   Rain      |  Normal  | Weak     | Yes  |<br>
|   Rain      |    Low   | Weak     | Yes  |<br>
|   Rain      |    Low   | Strong   | No   |<br>
|  Overcast   |    Low   | Strong   | Yes  |<br>
|   Hot       |  Normal  | Weak     | No   |<br>
|   Hot       |    Low   | Weak     | Yes  |<br>
|   Rain      |  Normal  | Weak     | Yes  |<br>
|   Hot       |  Normal  | Strong   | Yes  |<br>
|  Overcast   |  Normal  | Strong   | Yes  |<br>
|  Overcast   |   High   | Weak     | Yes  |<br>
|   Rain      |  Normal  | Strong   | No   |<br>
+-------------+----------+----------+------+


