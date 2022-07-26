"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from placement.pipelines import data_preprocessing as dp, model_training as mt
import os


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_preprocessing_pipeline = dp.create_pipeline()
    model_training_pipeline = mt.create_pipeline()
    # deploy = input("DO you want to deploy the model? (Y/N)")

    # if deploy == 'Y':
    #     os.system("mlflow models serve -m runs:/7dea1b2c5e8e4f528cbe132892afe078/model --port 1234")


    value = {"dp": data_preprocessing_pipeline, "mt":model_training_pipeline, "__default__":data_preprocessing_pipeline + model_training_pipeline}
    
    return value
