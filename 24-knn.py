import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Wczytywanie danych Iris
iris = load_iris()
X = iris.data
y = iris.target

# Skalowanie cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie i trenowanie modelu k-NN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Przewidywanie wartości dla zbioru testowego
y_pred = knn.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność: {accuracy:.2f}")

# Szczegółowy raport klasyfikacji
print(classification_report(y_test, y_pred))




## k-NN rekomendacje

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Przykładowe dane: oceny filmów przez użytkowników
data = {'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4],
        'rating': [5, 4, 3, 4, 5, 2, 5, 4, 3]}
df = pd.DataFrame(data)

# Przekształcenie danych na macierz użytkownik-film
ratings_matrix = df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Standard scaling
scaler = StandardScaler()
scaled_ratings = scaler.fit_transform(ratings_matrix)

# Model k-NN dla rekomendacji
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(scaled_ratings)

import numpy as np

# Wybór użytkownika, dla którego robimy rekomendacje (np. user_id = 1)
user_id = 1
user_index = user_id - 1  # indeks w macierzy (pandas index starts from 0)

# Znalezienie najbliższych sąsiadów
distances, indices = knn.kneighbors([scaled_ratings[user_index]], n_neighbors=3)

# Użytkownicy-sąsiedzi (z wyłączeniem samego siebie)
similar_users = indices.flatten()[1:] 

# Rekomendacje na podstawie ocen użytkowników-sąsiadów
recommended_movies = []

for user in similar_users:
    similar_user_id = user + 1  # konwersja z indeksu na user_id
    user_ratings = df[df['user_id'] == similar_user_id]
    high_rated_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id'].tolist()
    recommended_movies.extend(high_rated_movies)

# Usunięcie duplikatów i przekształcenie do listy
recommended_movies = list(set(recommended_movies))

print(f"Rekomendowane filmy dla użytkownika {user_id}: {recommended_movies}")



##  Ewaluacja

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# wczytanie modułów

# Wczytywanie danych Iris
iris = load_iris()
X = iris.data
y = iris.target

# Skalowanie cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie i trenowanie modelu k-NN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Przewidywanie wartości dla zbioru testowego
y_pred = knn.predict(X_test)

# Obliczanie metryk ewaluacyjnych
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Dokładność: {accuracy:.2f}")
print(f"Macierz konfuzji:\n{conf_matrix}")
print(f"Precyzja: {precision:.2f}")
print(f"Czułość: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
# Szczegółowy raport klasyfikacji 
print(classification_report(y_test, y_pred))

# Dla zbiorów wieloklasowych AUC-ROC wymaga zastosowania binarnej klasyfikacji wieloklasowej
# Tutaj przykładowo dla jednej klasy
if len(np.unique(y)) == 2:  # Sprawdzenie, czy mamy do czynienia z problemem binarnym
    y_proba = knn.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"AUC-ROC: {roc_auc:.2f}")

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

