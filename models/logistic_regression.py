from optbinning import BinningProcess

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class LogisticRegressionClf:
    def __init__(self):
        special_codes = [-9, -8, -7]

        variable_names = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
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

        self.lr = Pipeline(steps=[('binning_process', binning_process),
                        ('classifier', LogisticRegression(solver='lbfgs'))])
        self.model_name = 'Logistic Regression'

    def fit(self, X_train, y_train):
        self.lr.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.lr.predict(X_test)
        return y_pred

    def get_model(self):
        return self.lr

    def get_model_name(self):
        return self.model_name
