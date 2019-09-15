# -*- coding: utf-8 -*-
import boto3
import yaml

import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.model import Model

with open('config.yaml', "r") as f:
    config = yaml.load(f)

sess = sagemaker.Session()
role = config['role']
model_data = config['model_data']

if __name__ == '__main__':
    container = get_image_uri(
        boto3.Session().region_name,
        'xgboost'
    )

    xgb = Model(
        model_data=model_data,
        image=container,
        role=role,
        sagemaker_session=sess
    )

    xgb_predictor = xgb.deploy(
        initial_instance_count=1,
        instance_type='ml.m4.xlarge'
    )
