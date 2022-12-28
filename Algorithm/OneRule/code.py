"""
created by: aek426rahmani
created on: 28-12-2022
"""
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