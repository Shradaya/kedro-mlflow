"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from placement.pipelines import data_preprocessing as dp, model_training as mt


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    # data_preprocessing_pipeline = dp.create_pipeline()
    model_training_pipeline = mt.create_pipeline()

    value = {"dp": 'q', "mt":model_training_pipeline, "__default__":model_training_pipeline}
    
    return value
