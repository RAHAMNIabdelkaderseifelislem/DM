## One Rule Algorithm
is a supervised algorithm that's used to classify data into two or more classes. It's based on the idea that a single rule can be used to classify the data. This rule is called the **decision rule**. It's a rule that classifies a data instance into one of the classes.

The algorithm is as follows:

1. For each attribute, calculate the error rate.
2. Choose the attribute with the lowest error rate.
3. The decision rule is the attribute with the lowest error rate.
### Example
Let's assume we have the following data:
<code>
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
</code>

<br>
now we can calculate the error rate for each attribute:
<br>
<br> the error rate for the attribute **Temperature** is:
<br>
$$
\frac{3}{5} = 0.6
$$ + 
$$
\frac{2}{9} = 0.222
$$ =
$$
0.822
$$

