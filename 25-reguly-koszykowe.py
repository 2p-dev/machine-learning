import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Przykładowe dane transakcyjne
transactions = [['mleko', 'chleb', 'masło'],
                ['chleb', 'masło'],
                ['mleko', 'chleb'],
                ['mleko', 'chleb', 'masło'],
                ['chleb', 'masło']]

# Przekształcenie danych do formatu odpowiedniego dla algorytmu
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Znalezienie częstych zbiorów przedmiotów
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Generowanie reguł asocjacyjnych
rules = association_rules(frequent_itemsets,  metric='confidence',min_threshold=0.7, num_itemsets=len(transactions))


print(frequent_itemsets)
print(rules)



## Przykład 2

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Przykładowe dane transakcyjne
transactions = [
    ['mleko', 'chleb', 'masło'],
    ['chleb', 'masło'],
    ['mleko', 'chleb'],
    ['mleko', 'chleb', 'masło'],
    ['chleb', 'masło'],
    ['masło', 'jajka'],
    ['mleko', 'jajka'],
    ['jajka', 'chleb'],
    ['mleko', 'chleb', 'jajka', 'masło'],
    ['chleb', 'masło', 'jajka']
]

# Przekształcenie danych do formatu odpowiedniego dla algorytmu
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)


# Znalezienie częstych zbiorów przedmiotów
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print(frequent_itemsets)


# Generowanie reguł asocjacyjnych
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=len(transactions))
print(rules)


# Funkcja do rekomendacji produktów na podstawie reguł asocjacyjnych
def recommend_products(rules, product, top_n=3):
    # Filtrujemy reguły, gdzie produkt jest w antecedents (poprzednikach)
    recommendations = rules[rules['antecedents'].apply(lambda x: product in x)]
    recommendations = recommendations.sort_values(by='confidence', ascending=False)
    
    # Wyciągamy rekomendowane produkty
    recommended_products = []
    for _, row in recommendations.iterrows():
        consequent_products = list(row['consequents'])
        recommended_products.extend(consequent_products)
    
    recommended_products = list(set(recommended_products))
    
    # Zwracamy top_n rekomendowanych produktów
    return recommended_products[:top_n]

# Przykład: rekomendacja produktów dla "mleko"
product = 'mleko'
recommended_products = recommend_products(rules, product, top_n=3)
print(f"Rekomendowane produkty dla '{product}': {recommended_products}")


