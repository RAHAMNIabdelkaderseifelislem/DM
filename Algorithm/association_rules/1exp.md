## Association Rules Algorithm
[]: # 
[]: # Association rules are a type of rule that is used in data mining to find interesting relationships between variables in large databases. The interestingness of a rule is measured by a measure called support, confidence, and lift.
[]: # 
[]: # The algorithm is as follows:
[]: # 
[]: # 1. Set a minimum support and confidence.
[]: # 2. Take all the subsets in transactions having higher support than minimum support.
[]: # 3. Take all the rules of these subsets having higher confidence than minimum confidence.
[]: # 4. Sort the rules by decreasing lift.
[]: # 
[]: # ### Example
[]: # 
[]: # Let's say