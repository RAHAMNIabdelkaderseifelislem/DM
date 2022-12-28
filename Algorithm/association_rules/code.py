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

