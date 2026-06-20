import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample Dataset
data = {
    'Milk': [1, 1, 0, 1, 0],
    'Bread': [1, 1, 1, 0, 1],
    'Butter': [0, 1, 1, 1, 0],
    'Eggs': [1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:\n")
print(df)

# Apply Apriori Algorithm
frequent_items = apriori(df, min_support=0.4, use_colnames=True)

print("\nFrequent Itemsets:\n")
print(frequent_items)

# Generate Association Rules
rules = association_rules(frequent_items,
                          metric="confidence",
                          min_threshold=0.5)

print("\nAssociation Rules:\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])