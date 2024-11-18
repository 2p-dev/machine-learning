import pandas as pd

# wczytanie pliku
df = pd.read_excel("przyklad-pandas.xlsx", sheet_name="Sheet1")

# wypisanie podstawowych informacji o DataFrame
print(df.describe())
print(df)

# wstawienie 'BRAK' do kolumny imie jesli jest puste
df['imie'].fillna('BRAK', inplace=True)

# Obliczenie średniej dla kolumny wartosc
m = df['wartosc'].mean()
# Uzupełnianie brakujących wartości średnią kolumn
df['wartosc'].fillna(m, inplace=True)  

# Usuwanie wszystkich pozostałych wierszy, gdzie występują puste wartości
df.dropna(inplace=True)

# Usunięcie wszystkich wartości 'ekstremalnych', poza 3*odchyleniem standardowym
m = df['wartosc'].mean()
std = df['wartosc'].std()
print('srednia:',m,'odchylenie standardowe:', std)
prog = 3
outliers = df[(df['wartosc'] < m - prog * std) | (df['wartosc'] > m + prog * std)]
# nadpisanie poprzedniej df tylko danymi które są w zakresie
df = df[(df['wartosc'] >= m - prog * std) & (df['wartosc'] <= m + prog * std)]

# przekształcenie kolumny tekstowej na wartości liczbowe
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['kurier_encoded'] = label_encoder.fit_transform(df['kurier'])

print(df)
