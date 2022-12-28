## Association Rules Algorithm

Association rules are a type of rule-based machine learning method for discovering interesting relations between variables in large databases. They are used in many fields including market basket analysis in retail, in recommender systems, and in bioinformatics.

The algorithm is as follows:

1. Set a minimum support and confidence.
2. Take all the subsets in transactions having higher support than minimum support.
3. Take all the rules of these subsets having higher confidence than minimum confidence.
4. Sort the rules by decreasing lift.

### Example

Let's say we have the following data:

| id | item |
|----|------|
| 1  | A    |
| 1  | B    |
| 1  | C    |
| 2  | A    |
| 2  | B    |
| 3  | B    |
| 3  | C    |
| 4  | A    |
| 4  | C    |
| 5  | B    |
| 5  | C    |

We want to find the association rules with a minimum support of 50% and a minimum confidence of 75%.

1. Set a minimum support and confidence.
2. Take all the subsets in transactions having higher support than minimum support.
3. Take all the rules of these subsets having higher confidence than minimum confidence.
4. Sort the rules by decreasing lift.

The subsets are:

| itemset | support |
|---------|---------|
| A       | 40%     |
| B       | 60%     |
| C       | 60%     |
| A, B    | 20%     |
| A, C    | 20%     |
| B, C    | 40%     |
| A, B, C | 20%     |

The rules are:

| antecedent | consequent | support | confidence |
|------------|------------|---------|------------|
| A          | B          | 20%     | 50%        |
| B          | A          | 20%     | 33%        |
| A          | C          | 20%     | 50%        |
| C          | A          | 20%     | 33%        |
| B          | C          | 40%     | 67%        |
| C          | B          | 40%     | 67%        |

The rules sorted by decreasing lift are:

| antecedent | consequent | support | confidence | lift |
|------------|------------|---------|------------|------|
| B          | C          | 40%     | 67%        | 1.12 |
| C          | B          | 40%     | 67%        | 1.12 |
| A          | B          | 20%     | 50%        | 0.83 |
| B          | A          | 20%     | 33%        | 0.55 |
| A          | C          | 20%     | 50%        | 0.83 |
| C          | A          | 20%     | 33%        | 0.55 |

## Implementation

```python
import numpy as np
import pandas as pd
from apyori import apriori

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []

for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

results = list(rules)

print(results)
```

result is:

```python
[RelationRecord(items=frozenset({'light cream', 'chicken'}), support=0.004532728969470737, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)]), RelationRecord(items=frozenset({'escalope', 'mushroom cream sauce'}), support=0.005732568990801226, ordered_statistics=[OrderedStatistic(items_base=frozenset({'mushroom cream sauce'}), items_add=frozenset({'escalope'}), confidence=0.3006993006993007, lift=3.790832696715049)])]
```

