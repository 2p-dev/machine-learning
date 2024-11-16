import pandas as pd
# Returns a DataFrame
xls = pd.read_excel("przyklad-pandas.xlsx", sheet_name="Sheet1")
print(xls)

#    id   wartosc     imie
# 0   1      3.50      Jan
# 1   2      2.80     Adam
# 2   3      1.10    Zofia
# 3   4      3.14  Grażyna
# 4   5  88124.00  Tadeusz

xls = pd.read_excel("przyklad-pandas.xlsx", sheet_name="Sheet2", 
                    header=None, names=['numer', 'waga', 'imie'])
print(xls)


xls.to_json('przyklad-pandas.json')

xls = pd.read_excel("przyklad-pandas.xlsx", sheet_name="Sheet1")
xls_wyczyszczony = xls.dropna()
print(xls_wyczyszczony) # usuwa wiersz z brakującymi danymi, nie zmienia indeksownia wierszy

xls_zamieniony_0 = xls.fillna(0)
print(xls_zamieniony_0)

slownik_domyslnych_wartosci = {'id': 0, 'wartosc': 5.0, 'imie': 'X'}
xls_zamieniony_slownikiem = xls.fillna(slownik_domyslnych_wartosci)
print(xls_zamieniony_slownikiem)

# informacje / operacje 


data = xls_zamieniony_slownikiem

# Podstawowe operacje na danych
print(data.describe())  # Statystyki opisowe

#              id       wartosc
# count  6.000000      6.000000 # liczba niepustych wartości
# mean   2.500000  14689.923333 # wartość średnia
# std    1.870829  35975.203535 # odchylenie standardowe
# min    0.000000      1.100000 # minimum
# 25%    1.250000      2.885000
# 50%    2.500000      3.320000 # mediana
# 75%    3.750000      4.625000 
# max    5.000000  88124.000000 # maksimum


filtrowane = data[data['wartosc'] >= 5.0]
print(filtrowane)

pogrupowane = data.groupby('imie')
print(pogrupowane.count())