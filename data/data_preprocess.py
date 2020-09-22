import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/raw/heloc_dataset_v1.csv')

variable_names = list(df.columns[1:])

X = df[variable_names].values

y = df.RiskPerformance.values
mask = y == "Bad"
y[mask] = 1
y[~mask] = 0
y = y.astype(np.int)

train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv('data/preprocessed/train.csv', index=False)
test.to_csv('data/preprocessed/test.csv', index=False)
