from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
import boto3
import os


class copyHooks:
    @hook_impl
    def before_pipeline_run(self, catalog: DataCatalog) -> None:
        # s3 = boto3.resource('s3')
        # s3.meta.client.upload_file('/data/06_models/lgb_model', 'shradaya', 'lgb_model')
        pass