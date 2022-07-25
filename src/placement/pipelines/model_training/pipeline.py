"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_training_tracking, deploy_model, tune_hyperparameters


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
                    [node(tune_hyperparameters,
                        ['X_train', 'y_train', 'X_valid', 'y_valid'],
                        "auc",
                        name = 'train')]
                    )
