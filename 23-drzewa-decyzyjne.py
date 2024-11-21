from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

# Wczytywanie danych Iris
iris = load_iris()
X = iris.data
y = iris.target

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie i trenowanie drzewa decyzyjnego
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Przewidywanie wartości dla zbioru testowego
y_pred = clf.predict(X_test)

# Obliczanie dokładności
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność: {accuracy:.2f}")

# Wizualizacja drzewa decyzyjnego
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()



## Random forest

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Wczytywanie danych Iris
iris = load_iris()
X = iris.data
y = iris.target

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie i trenowanie modelu Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Przewidywanie wartości dla zbioru testowego
y_pred = clf.predict(X_test)

# Obliczanie dokładności
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność: {accuracy:.2f}")

# Ocena istotności cech
importances = clf.feature_importances_
for feature, importance in zip(iris.feature_names, importances):
    print(f"{feature}: {importance:.2f}")


# projekt drzew decyzyjnych i random forest


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree



# Wczytywanie danych Iris
iris = load_iris()
X = iris.data
y = iris.target

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konwersja do DataFrame dla lepszej analizy
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Tworzenie i trenowanie drzewa decyzyjnego
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)

# Przewidywanie wartości dla zbioru testowego
y_pred_tree = clf_tree.predict(X_test)

# Ocena modelu
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Dokładność drzewa decyzyjnego: {accuracy_tree:.2f}")

# Macierz konfuzji i raport klasyfikacji
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

plt.figure(figsize=(12, 8))
tree.plot_tree(clf_tree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

# Tworzenie i trenowanie modelu Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

# Przewidywanie wartości dla zbioru testowego
y_pred_rf = clf_rf.predict(X_test)

# Ocena modelu
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Dokładność Random Forest: {accuracy_rf:.2f}")

# Macierz konfuzji i raport klasyfikacji
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


importances = clf_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title("Ważność cech")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), np.array(iris.feature_names)[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
