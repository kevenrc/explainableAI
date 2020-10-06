import pandas as pd
import numpy as np

from optbinning import BinningProcess

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

raw_data = pd.read_csv('data/raw/heloc_dataset_v1.csv', delimiter=',')
variable_names = list(raw_data.columns[1:])
X_train = np.genfromtxt('data/preprocessed/X_train.csv', delimiter=',')
X_test = np.genfromtxt('data/preprocessed/X_test.csv', delimiter=',')
y_train = np.genfromtxt('data/preprocessed/y_train.csv', delimiter=',')
y_test = np.genfromtxt('data/preprocessed/y_test.csv', delimiter=',')

special_codes = [-9, -8, -7]

binning_fit_params = {
    "ExternalRiskEstimate": {"monotonic_trend": "descending"},
    "MSinceOldestTradeOpen": {"monotonic_trend": "descending"},
    "MSinceMostRecentTradeOpen": {"monotonic_trend": "descending"},
    "AverageMInFile": {"monotonic_trend": "descending"},
    "NumSatisfactoryTrades": {"monotonic_trend": "descending"},
    "NumTrades60Ever2DerogPubRec": {"monotonic_trend": "ascending"},
    "NumTrades90Ever2DerogPubRec": {"monotonic_trend": "ascending"},
    "PercentTradesNeverDelq": {"monotonic_trend": "descending"},
    "MSinceMostRecentDelq": {"monotonic_trend": "descending"},
    "NumTradesOpeninLast12M": {"monotonic_trend": "ascending"},
    "MSinceMostRecentInqexcl7days": {"monotonic_trend": "descending"},
    "NumInqLast6M": {"monotonic_trend": "ascending"},
    "NumInqLast6Mexcl7days": {"monotonic_trend": "ascending"},
    "NetFractionRevolvingBurden": {"monotonic_trend": "ascending"},
    "NetFractionInstallBurden": {"monotonic_trend": "ascending"},
    "NumBank2NatlTradesWHighUtilization": {"monotonic_trend": "ascending"}
}

binning_process = BinningProcess(variable_names, special_codes=special_codes,
        binning_fit_params=binning_fit_params)

clf_lr = Pipeline(steps=[('binning_process', binning_process),
                        ('classifier', LogisticRegression(solver='lbfgs'))])

clf_lr.fit(X_train, y_train)
y_pred = clf_lr.predict(X_test)
print(classification_report(y_test, y_pred))
