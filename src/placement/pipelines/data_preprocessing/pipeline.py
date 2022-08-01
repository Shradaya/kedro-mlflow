"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import encode_target_col, exclude_feature, encode_categorical_features, train_test_spliter

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(encode_target_col,
            ['raw_data', 'params:target_col'],
            ['data', 'target'],
            name = 'separate_target_and_encode'),
        node(exclude_feature,
            ['data', 'params:drop_features'],
            'prefered_data',
            name = 'exclude_Features'),
        node(encode_categorical_features,
            ['prefered_data'],
            'encoded_data',
            name = 'encode'),
        node(train_test_spliter,
            ['encoded_data', 'target'],
            ['X_train', 'X_valid', 'y_train', 'y_valid'],
            name = 'split')
    ])
