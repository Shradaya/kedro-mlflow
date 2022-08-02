"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import deploy_model, tune_hyperparameters, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
                    [node(tune_hyperparameters,
                        ['X_train', 'y_train', 'X_valid', 'y_valid'],
                        'best_param',
                        name = 'tune'),
                    node(train_model,
                        ['best_param', 'X_train', 'y_train', 'X_valid', 'y_valid'],
                        'lightgbm_model',
                        name = 'train'),
                    # node(deploy_model,
                    #     'params:model_path',
                    #     "sth",
                    #     name = 'deploy')
                    ]
                    )
