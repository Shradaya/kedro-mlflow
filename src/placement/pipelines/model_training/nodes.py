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
    # experiment_id = create_experiment("campus_recruitment_experiments")
    with mlflow.start_run(nested=True) as run:
        run_id = run.info.run_uuid
        run_name = run_id  # used to provide a name to the run, helps while comparing different runs
        run_id_list.append(run_id)
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
        print("=====================================")   

        # log metrics
        mlflow.log_metrics({"Validation_AUC": roc_auc})
        return roc_auc, [run_id,]

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
