import numpy as np
import streamlit as st
import altair as alt
from sklearn.model_selection import train_test_split
import argparse
import os

from models.logistic_regression import LogisticRegressionClf
from models.xgboost import XGBoostClf
from models.neuralnet import NeuralNet
from scripts.explain.shap import Shap
from util.performances import *
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Explainable AI')

important_features = ["ExternalRiskEstimate", "MSinceMostRecentInqexcl7days", "AverageMInFile", "NetFractionRevolvingBurden", "PercentTradesNeverDelq", "NumSatisfactoryTrades"]

readable_feature_names = {
        "ExternalRiskEstimate":"Considated Risk Measures",
        "MSinceMostRecentInqexcl7days":"Months Since Most Recent Inquiry",
        "AverageMInFile":"Average Months in File",
        "NetFractionRevolvingBurden":"% Used Revolving Credit",
        "PercentTradesNeverDelq":"% Trades Never Delq",
        "NumSatisfactoryTrades":"Number Satisfactory Trades"}

#@st.cache
def load_data(path):
    data = pd.read_csv(path)
    return data

#@st.cache
def preprocess_data(data, label_name, labels=["Good", "Bad"]):

    X = data.drop(label_name, axis=1)
    y = data[label_name]
    mask = y == labels[1]
    y[mask] = 1
    y[~mask] = 0
    y = y.astype(np.int)

    return X, y

#@st.cache
def split_test_train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#@st.cache
def build_model(model_selected, X_train, y_train):

    model = None

    if model_selected == 'Logistic Regression':
        model = LogisticRegressionClf()

    if model_selected == 'XGBoost':
        model = XGBoostClf()

    if model_selected == 'Neural Net':
        model = NeuralNet()

    model.fit(X_train, y_train)

    return model, model.get_model_name()

def update_feature_value(data):
    for k, v in data.items():
        if k not in important_features:
            continue
        new_v = st.sidebar.number_input("{0}".format(readable_feature_names[k]),
                                  min_value=int(-9),
                                  max_value=int(100),
                                  value = int(v))
        if new_v:
            data[k] = new_v
    return data

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        action="store",
        default=os.getcwd() + '/data/raw/heloc_dataset_v1.csv',
        help="dataset path",
    )

    parser.add_argument(
        "--label_name",
        action="store",
        default="RiskPerformance",
        help="Name of Label column"
    )

    args = parser.parse_args()

    # load data
    df_raw = load_data(args.data)
    label_name = args.label_name

    # preprocess
    X, y = preprocess_data(df_raw, label_name)
    X_train, X_test, y_train, y_test = split_test_train(X, y)

    # streamlit
    st.sidebar.markdown('# Model')

    # select model
    model_selected = st.sidebar.selectbox(
        'Select model',
        ('Logistic Regression', 'XGBoost', 'Neural Net'),
        index=0
    )

    # model
    model, title = build_model(model_selected, X_train, y_train)

    # prediction
    y_pred = predict_labels(model, X_test, y_test)

    # get performace
    performance = get_performances(y_test, y_pred)

#    st.subheader('model performance')
#    st.table(performance)

    # index input
    max_index = X_test.shape[0] - 1
    index = st.sidebar.number_input("Select the user index from 0 to {}: ".format(max_index),
                                    value=10,
                                    min_value=0,
                                    max_value=max_index,
                                    format="%d")

    # user data
    data = X_test.iloc[index, :].copy()

    # st.table(data)

    data = update_feature_value(data)

    # explain button
    explain_button = st.sidebar.button('Explanation')

    predict_button = st.sidebar.button('Predict')

    if predict_button:
        what_if_pred = predict_labels(model, data.to_frame().T, y_test)
        print(type(what_if_pred))

        if what_if_pred[0] == 1:
            st.markdown('Prediction: Bad')
        else:
            st.markdown('Prediction: Good')

    if explain_button:

        indexes = list(data.index)
        user_pred = y_pred.iloc[index]

        if user_pred == 1:
            st.markdown('As is: Bad')
        else:
            st.markdown('As is: Good')

        st.write("")

        background_data = X_train[:5000].copy()
        explain_data = X_test[:500].copy()

        # explain summary
        shap = Shap(title, model.get_model(), background_data)
        shap.explain(explain_data)
        shap_values = shap.get_shap_values()

        # visualization
        st.subheader("What is affecting my score most?")
        user_shap_values = shap_values[index]
        pivot = np.sum(np.abs(user_shap_values))
        user_shap_prob = np.round(user_shap_values / pivot, 2)
        user_shap_prob_df = pd.DataFrame({'index': indexes, 'probability': user_shap_prob})

        bars = alt.Chart(user_shap_prob_df).mark_bar().encode(
            x='index',
            y=alt.X('probability', axis=alt.Axis(format='%', title='Influence Factor')),
            color=alt.condition(
                alt.datum.probability >= 0,
                alt.value('#F63366'),
                alt.value('#1E88E5')
            )
        ).properties(width=640, height=480)

        text_above = bars.transform_filter(alt.datum.probability >= 0).mark_text(
            align='center',
            baseline='bottom',
            dy=-3
        ).encode(text='probability')

        text_below = bars.transform_filter(alt.datum.probability < 0).mark_text(
            align='center',
            baseline='bottom',
            dy=15
        ).encode(text='probability')

        st.altair_chart(bars + text_above + text_below)

        # summary plot
        st.subheader("What effects Score in general?")
        fig_summary = shap.plot_summary(explain_data)
        st.pyplot(fig_summary, bbox_inches='tight', dpi=300, pad_inches=0)

        # force plot
        st.subheader("Local Interpretation")
        fig_force = shap.plot_force(data, index)
        st.pyplot(fig_force, bbox_inches='tight', dpi=300, pad_inches=0)
if __name__ == '__main__':
    main()
