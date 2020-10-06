import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

def get_performances(y, y_pred):
    accuracy = np.round(metrics.accuracy_score(y, y_pred), 4)

    precision = np.round(metrics.precision_score(y, y_pred, average='weighted'), 4)

    recall = np.round(metrics.recall_score(y, y_pred, average='weighted'), 4)

    f1 = np.round(metrics.f1_score(y, y_pred, average='weighted'), 4)

    df = pd.DataFrame([[accuracy, precision, recall, f1]], index=['performance'], columns=['accuracy', 'precision', 'recall', 'f1'])
    return df

def get_classification_report(y, y_pred, labels):
    report = metrics.classification_report(y, y_pred, labels=labels)

    return report

def get_confusion_matrix(y, y_pred, labels):
    total_classes = len(classes)
    level_labels = [total_classes * [0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y, y_pred, labels=labels)

    cm_frame = pd.DataFrame(data=cm, columns=pd.MultiIndex(levels=[['Predicted:'], labels], labels=level_labels),
                            index=pd.MultiIndex(levels=[['Actual:'], labels], labels=level_labels))
    return cm_frame

def display_model_performance_metrics(y, y_pred, labels):
    print('Model Performance metrics:')
    print('-' * 30)
    get_metrics(y, y_pred, labels)
    print('\nModel Classification report:')
    print('-' * 30)
    display_classification_report(y, y_pred, lables)
    print('\nPrediction Confusion Matrix:')
    print('-' * 30)
    get_confusion_matrix(y, y_pred, labels)

def predict_labels(model, X, y):
    y_pred = model.predict(X)
    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)
    y_pred = pd.Series(y_pred.ravel())
    return y_pred
