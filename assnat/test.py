import pandas as pd
import numpy as np

# Exemple de Series
series = pd.Series([1, 2, 3])

# Exemple de numpy.ndarray
array = np.array([4, 5, 6])

# Vérification des longueurs
if len(series) == len(array):
    # Concaténation en tant que nouvelles colonnes dans un DataFrame
    df = pd.concat([series, pd.Series(array)], axis=1)
    df.columns = ['Col_Series', 'Col_Array']  # Nommer les colonnes
else:
    raise ValueError("Les longueurs de la Series et du ndarray ne correspondent pas.")

print(df)
