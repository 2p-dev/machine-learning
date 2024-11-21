# import podstawowych modułów
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# aby otrzymać te same wyniki losowe można zainicjować moduł losowy konkretną wartością
np.random.seed(0) 

# funkcja fit potrzebuje jako pierwszy parametr - tablicę tablic [[],[],[]] 
# zatem aby wygenerować taką tablicę losową można albo skorzystać z parametrów funcji rand(wymiar0, wymiar1, ...)
# albo skorzystać z metody reshape() na liście jednowymiarowej
# np.: np.random.rand(100).reshape(100,-1)
X = np.random.rand(100, 1)  # Zmienna niezależna 

# Tworzymy sobie zmienną zależną, na podstawie równania y = 2.5 * x + 8, 
# dodajemy losowy błąd, który pozwoli nam "odsunąć" dane od równania

y = 2.5 * X + 8 + np.random.randn(100, 1) / 5  # Zmienna zależna + "błąd"

# Utworzenie modelu regresji liniowej i dopasowanie
model = LinearRegression()
model.fit(X, y)

# Wartości przewidywane przez model regresji liniowej
y_pred = model.predict(X)  

# przykładowa ewaluacja wyników modelu

r2 = r2_score(y, y_pred) # Obliczenie R-kwadrat.
print("R kwadrat:", r2)

# Współczynnik a w równaniu y = a * x + b
a = model.coef_
print("Współczynnik a (nachylenie / slope):", a)  # Współczynnik b w równaniu y = a * x + b
b = model.intercept_
print("Współczynnik b:", b)

# przedstawienie danych i linii regresji na modelu
plt.scatter(X, y, color='blue', label='Dane')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regresja liniowa')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()



### Regresja liniowa wielokrotna


X1 = np.random.rand(100, 1)
X2 = np.random.rand(100, 1)
X = np.concatenate((X1, X2), axis=1)
y = -3 * X1 + 4 * X2 + 8 + np.random.randn(100, 1) / 10  

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)  

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("MSE: ", mse, "R2: ", r2)

# Współczynniki a i b w równaniu y = a * x1 + b * x2 + c
ab = model.coef_
print("Współczynniki a i b (nachylenie / slope):", ab)  
c = model.intercept_
print("Współczynnik c:", c)


# wyświetlanie 
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = [np.amin(X[:, 0]), np.amax(X[:, 0])]
ys = [np.amin(X[:, 1]), np.amax(X[:, 1])]

ax.scatter(X[:, 0], X[:, 1], y[:, 0], c='blue') # punkty danych

xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
zz = ab[0, 0] * xx + ab[0, 1] * yy + c
xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
ax.plot_surface(xx, yy, zz, color='red', alpha=0.5) # płaszczyzna wynikowa

ax.set_xlabel('X[:, 0] feature 0')
ax.set_ylabel('X[:, 1] feature 1')
ax.set_zlabel('Y')
plt.show()


# ewaluacja wyników modelu

import numpy as np
from sklearn.metrics import mean_squared_error

# Przykładowe dane rzeczywiste (y) i przewidywane (y_pred)
y = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Obliczanie R2 i MSE
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("MSE: ", mse, "R2: ", r2)


# projekt

# mając dane zmienne: niezależną i zależną wytrenuj model wykorzystując regresję liniową
# wypisz współczynniki dopasowanej krzywej regresji
# wypisz miary R2 i MSE. 
# oblicz oblicz predykcję dla wartości zmiennych zależnych X_test

X_arr = [-10, -2, 0, 9, 12, 34, 74, 123]
y_arr = [ 13, 0, 2, -20,  -25, -60, -90, -150]
X_test = [-100, 30, 200]


X = np.array(X_arr).reshape(8, -1)
y = np.array(y_arr).reshape(8, -1)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)  

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("MSE: ", mse, "R2: ", r2)

# Współczynnik a w równaniu y = a * x + b
a = model.coef_
print("Współczynnik a (nachylenie / slope):", a)  # Współczynnik b w równaniu y = a * x + b
b = model.intercept_
print("Współczynnik b:", b)

# przedstawienie danych i linii regresji na modelu
plt.scatter(X, y, color='blue', label='Dane')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regresja liniowa')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

y_pred = model.predict(np.array(X_test).reshape(3, -1))  
print (y_pred)
