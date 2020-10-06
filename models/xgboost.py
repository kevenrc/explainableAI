import xgboost as xgb
from xgboost import XGBClassifier

class XGBoostClf:
    def __init__(self):
        self.params = {
        # Parameters that we are going to tune.
        'max_depth':5,
        'min_child_weight': 8,
        'eta':.1,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        # Other parameters
        'objective':'reg:squarederror',
        }
        self.params['eval_metric'] = "error"
        self.num_boost_round = 999
        self.xgclf = XGBClassifier(**self.params)
        self.model_name = "XGBoost"

    def fit(self, X_train, y_train):
        self.xgclf.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.xgclf.predict(X_test)
        return y_pred

    def get_model(self):
        return self.xgclf

    def get_model_name(self):
        return self.model_name
