import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/raw/heloc_dataset_v1.csv')

variable_names = list(df.columns[1:])
print(df.describe())

X = df[variable_names].values

y = df.RiskPerformance.values
mask = y == "Bad"
y[mask] = 1
y[~mask] = 0
y = y.astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#np.savetxt('X_train.csv', X_train, delimiter=',')
#np.savetxt('X_test.csv', X_test, delimiter=',')
#np.savetxt('y_train.csv', y_train, delimiter=',')
#np.savetxt('y_test.csv', y_test, delimiter=',')
