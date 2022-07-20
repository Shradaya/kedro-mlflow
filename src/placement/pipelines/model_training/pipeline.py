"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_training_tracking, prepare_hyperparameters


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
                    [node(prepare_hyperparameters,
                        ['params:learning_rate', 'params:colsample_bytree', 'params:subsample'],
                        'parameter',
                        name = 'hpprep'),
                    node(model_training_tracking,
                        ['parameter', 'X_train', 'y_train', 'X_valid', 'y_valid'],
                        ['auc', 'run_id'],
                        name = 'train')]
                    )
