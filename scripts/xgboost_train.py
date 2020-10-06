import pandas as pd
import numpy as np

import xgboost as xgb
from optbinning import BinningProcess
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

X_train = np.genfromtxt('data/preprocessed/X_train.csv', delimiter=',')
X_test = np.genfromtxt('data/preprocessed/X_test.csv', delimiter=',')
y_train = np.genfromtxt('data/preprocessed/y_train.csv', delimiter=',')
y_test = np.genfromtxt('data/preprocessed/y_test.csv', delimiter=',')

params = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 8,
    'eta':.1,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    # Other parameters
    'objective':'reg:squarederror',
}
params['eval_metric'] = "error"
num_boost_round = 999

clf_xgb = XGBClassifier(**params)
model = clf_xgb.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
