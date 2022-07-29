# Data analysis library
import numpy as np
import pandas as pd
import joblib

# Machine Learning library
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Model experimentation library
import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import optuna
from functools import partial
# Hyperparameter tunning library

import warnings
import os
warnings.filterwarnings("ignore")

def model_training_tracking(params, X_train, y_train, X_valid, y_valid):
    lgb_clf = LGBMClassifier(**params)
    lgb_clf.fit(X_train, y_train, 
                eval_set = [(X_train, y_train), (X_valid, y_valid)], 
                early_stopping_rounds=50,
                verbose=20)

    mlflow.sklearn.log_model(lgb_clf, "model")
    
    lgb_valid_prediction = lgb_clf.predict_proba(X_valid)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, lgb_valid_prediction)
    roc_auc = auc(fpr, tpr) # compute area under the curve
    print("=====================================")
    print("Validation AUC:{}".format(roc_auc))
    auc_metric = {"Validation_AUC": roc_auc}
    mlflow.log_metrics(auc_metric)
    print(mlflow.active_run())
    print(f"Logged metrics {auc_metric}")
    print("=====================================")

    return roc_auc


def objective(X_train, y_train, X_valid, y_valid, trial):
    param = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1e-1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "random_state": 42,
    }
    trial_dict = {}
    trial_count = str(trial.__dict__['_trial_id'])
    trial_dict[f"trial_{trial_count}_learning_rate"]= param['learning_rate']
    trial_dict[f"trial_{trial_count}_colsample_bytree"]= param['colsample_bytree']
    trial_dict[f"trial_{trial_count}_subsample"]= param['subsample']
    mlflow.log_params(trial_dict)
    auc = model_training_tracking(param, X_train, y_train, X_valid, y_valid)
    return auc

def tune_hyperparameters(X_train, X_valid, y_test, y_valid):
    study = optuna.create_study(direction='maximize')
    fun_objective = partial(objective, X_train, X_valid, y_test, y_valid)
    mlflc = optuna.integration.MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=f"test_precision",
        nest_trials=True
    )
    study.optimize(fun_objective, n_trials=1, callbacks = [mlflc])

    trial = study.best_trial
    print('AUC: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    return trial

def deploy_model(model_path):
    model = mlflow.pyfunc.load_model(model_path)
    mlflow.register_model(f"file://{model_path}", "sample-sklearn-mlflow-model")
    return model_path