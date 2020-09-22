from fastai import *
import numpy as np

raw_data = pd.read_csv('data/raw/heloc_dataset_v1.csv', delimiter=',')
variable_names = list(raw_data.columns[1:])

X_train = np.genfromtxt('data/preprocessed/X_train.csv', delimiter=',')
X_test = np.genfromtxt('data/preprocessed/X_test.csv', delimiter=',')
y_train = np.genfromtxt('data/preprocessed/y_train.csv', delimiter=',')
y_test = np.genfromtxt('data/preprocessed/y_test.csv', delimiter=',')
