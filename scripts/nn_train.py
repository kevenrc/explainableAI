from fastai import *
import numpy as np

X_train = np.genfromtxt('data/preprocessed/X_train.csv', delimiter=',')
X_test = np.genfromtxt('data/preprocessed/X_test.csv', delimiter=',')
y_train = np.genfromtxt('data/preprocessed/y_train.csv', delimiter=',')
y_test = np.genfromtxt('data/preprocessed/y_test.csv', delimiter=',')
