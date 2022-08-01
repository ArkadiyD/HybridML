from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import joblib

'''
Code for training a random forest for predicting cherries
'''


def train_cherries_rf(X, Y, name=None):
    if name is None:
        model_name = f"RFModels/rf_cherries.joblib"
    else:
        model_name = f"RFModels/rf_cherries_{name}.joblib"

    # split data in train and validation
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

    # train model
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)

    # evaluation
    score_rf = rf.score(X_val, Y_val)
    score_rf_no_cher = rf.score(X_val[Y_val[0] == 1], Y_val[Y_val[0] == 1])
    score_rf_cher = rf.score(X_val[Y_val[1] == 1], Y_val[Y_val[1] == 1])
    score_rf_ret_cher = rf.score(X_val[Y_val[2] == 1], Y_val[Y_val[2] == 1])
    score_rf_no_ret_cher = rf.score(X_val[Y_val[3] == 1], Y_val[Y_val[3] == 1])

    print(f"\nRF overall validation accuracy = {score_rf}")
    print(f"RF no cherry validation accuracy = {score_rf_no_cher}")
    print(f"RF cherry validation accuracy = {score_rf_cher}")
    print(f"RF ret cherry validation accuracy = {score_rf_ret_cher}")
    print(f"RF no ret cherry validation accuracy = {score_rf_no_ret_cher}")

    # feature importance
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
    print("Feature importance:\n")
    print(feature_importance)

    # save
    joblib.dump(rf, model_name)

    return [score_rf, score_rf_no_cher, score_rf_cher, score_rf_ret_cher, score_rf_no_ret_cher]
