import pandas as pd
import numpy as np

s = pd.Series([1, 3, 5, np.nan, 6, 8])
# ponieważ np.nan jest float, to całe Series będzie zawierało liczby zmiennoprzecinkowe
print(s)

s_int = pd.Series([1, 3, 5, 6, 8], dtype="int32")
print(s_int)

daty = pd.date_range("20241116", periods=6)
print(daty)

losowe_dane = np.random.randn(6, 4)
print(losowe_dane)
df = pd.DataFrame(losowe_dane, index=daty, columns=list("ABCD"))
print(df)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html#pandas.Timestamp
df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp(2014, 11, 16), # konkretna data / czas
        "C": pd.Series(1, index=list(range(4)), dtype="float32"), # Series zawierająca 4 elementy o wartości 1.0
        "D": np.array([3] * 4, dtype="int32"), # tabela zawierająca 4 elementy o wartości 3
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "hej",
    }
)
#      A          B    C  D      E    F
# 0  1.0 2014-11-16  1.0  3   test  hej
# 1  1.0 2014-11-16  1.0  3  train  hej
# 2  1.0 2014-11-16  1.0  3   test  hej
# 3  1.0 2014-11-16  1.0  3  train  hej

print(df2)