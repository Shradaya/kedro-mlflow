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


def create_experiment(experiment_name):
    artifact_repository = './mlflow-run'
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')
    client = MlflowClient()
    try:
        experiment_id = client.create_experiment(experiment_name, artifact_location=artifact_repository)
    except:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    return experiment_id

def model_training_tracking(params, X_train, y_train, X_valid, y_valid):
    mlflow.log_params(params)
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

    return lgb_clf

def prepare_hyperparameters(learning_rate, colsample_bytree, subsample):
    param = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": learning_rate,
        "colsample_bytree": colsample_bytree,
        "subsample": subsample,
        "random_state": 42,
    }
    return param

def objective(X_train, y_train, X_valid, y_valid, trial):
    param = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1e-1, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "random_state": 42,
    }
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

def deploy_model(run_id):
    print(run_id)
    return run_id
