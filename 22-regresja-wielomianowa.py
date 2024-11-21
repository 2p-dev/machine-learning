
## Overfitting


import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset
np.random.seed(0)
X = np.linspace(0, 10, 40)
y = np.sin(X) + np.random.normal(scale=0.5, size=X.shape)

X = X[:, np.newaxis]

# Polynomial regression with degree 2 (underfitting)
poly_features_2 = PolynomialFeatures(degree=2)
X_poly_2 = poly_features_2.fit_transform(X)
model_2 = LinearRegression()
model_2.fit(X_poly_2, y)
y_poly_pred_2 = model_2.predict(X_poly_2)

# Polynomial regression with degree 15 (overfitting)
poly_features_15 = PolynomialFeatures(degree=15)
X_poly_15 = poly_features_15.fit_transform(X)
model_15 = LinearRegression()
model_15.fit(X_poly_15, y)
y_poly_pred_15 = model_15.predict(X_poly_15)

# Plotting the results
plt.scatter(X, y, color='gray', label='Data')
plt.plot(X, y_poly_pred_2, label='Stopień 2 (Underfitting)', color='blue')
plt.plot(X, y_poly_pred_15, label='Stopień 15 (Overfitting)', color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresja wielomianowa: Underfitting vs Overfitting')
plt.legend()
plt.show()

# Print MSE for both models
mse_2 = mean_squared_error(y, y_poly_pred_2)
mse_15 = mean_squared_error(y, y_poly_pred_15)
print(f"MSE dla stopnia 2: {mse_2:.2f}")
print(f"MSE dla stopnia 15: {mse_15:.2f}")


## Regresja

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generowanie przykładowych danych
np.random.seed(0)
X = 2 - 3 * np.random.normal(0, 1, 100)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 13, 100)

X = X[:, np.newaxis] # równoznaczne z X.reshape(-1, 1)
y = y[:, np.newaxis] # równoznaczne z X.reshape(-1, 1)

# Tworzenie cech wielomianowych
polynomial_features = PolynomialFeatures(degree=3, include_bias= False)
X_poly = polynomial_features.fit_transform(X)

# Tworzenie modelu regresji liniowej i dopasowywanie do danych
model = LinearRegression()
model.fit(X_poly, y)
y_poly_pred = model.predict(X_poly)

# Obliczanie błędu średniokwadratowego (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_poly_pred)
print("MSE:", mse)

# Wizualizacja wyników
plt.scatter(X, y, s=10)
# Sortowanie wartości dla wykresu
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, y_poly_pred), key=sort_axis)
X, y_poly_pred = zip(*sorted_zip)
plt.plot(X, y_poly_pred, color='m')
plt.show()




## Regresja - ćwiczenie

# mając dane zmienne: niezależną i zależną wytrenuj model wykorzystując regresję wielomianową
# wypisz współczynniki dopasowanej krzywej regresji
# wypisz miary R2 i MSE. 
# oblicz oblicz predykcję dla wartości zmiennych zależnych X_test
# spróbuj najpierw zwizualizować dane aby dopasować poziom wielomianu

X_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8,   9, 10, 11, 12, 13, 14]).reshape(-1, 1)
y_train = np.array([0, 1, 4, 6, 4, 2, 1, 0, -2, -6, -2,  0,  2, 8, 13]).reshape(-1, 1)
X_test = np.array([-10, 20])[:,np.newaxis]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generowanie przykładowych danych
np.random.seed(0)

# Tworzenie cech wielomianowych
polynomial_features = PolynomialFeatures(degree=3, include_bias= False)
X_poly = polynomial_features.fit_transform(X_train)

# Tworzenie modelu regresji liniowej i dopasowywanie do danych
model = LinearRegression()
model.fit(X_poly, y_train)
y_poly_pred = model.predict(X_poly)

X_test_poly = polynomial_features.transform(X_test)
print( model.predict(X_test_poly))

# Obliczanie błędu średniokwadratowego (MSE)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_train, y_poly_pred)
print("MSE:", mse)

# Wizualizacja wyników
plt.scatter(X_train, y_train, s=10)
# Sortowanie wartości dla wykresu
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_train, y_poly_pred), key=sort_axis)
X, y_poly_pred = zip(*sorted_zip)
plt.plot(X_train, y_poly_pred, color='m')
plt.show()
