# -*- coding: utf-8 -*-
import datetime

import boto3
import yaml

import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import s3_input

job_name = 'xgboost-churn-' + \
    str(datetime.datetime.now().strftime("%Y-%m-%d-%H%M"))

with open('config.yaml', "r") as f:
    config = yaml.load(f)

sess = sagemaker.Session()
role = config['role']
bucket = config['bucket']
key_prefix = config['key-prefix']
config['job-name'] = job_name

if __name__ == '__main__':
    input_train = sess.upload_data(
        path='train.csv',
        bucket=bucket,
        key_prefix=key_prefix
    )

    input_validation = sess.upload_data(
        path='validation.csv',
        bucket=bucket,
        key_prefix=key_prefix
    )

    s3_input_train = s3_input(
        s3_data=input_train,
        content_type='text/csv'
    )

    s3_input_validation = s3_input(
        s3_data=input_validation,
        content_type='text/csv'
    )

    container = get_image_uri(
        boto3.Session().region_name,
        'xgboost'
    )

    xgb = sagemaker.estimator.Estimator(
        container,
        role,
        train_instance_count=1,
        train_instance_type='ml.m4.xlarge',
        sagemaker_session=sess,
        output_path=config['s3_output_location']
    )

    xgb.set_hyperparameters(
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.8,
        silent=0,
        objective='binary:logistic',
        num_round=100
    )

    xgb.fit(
        inputs={
            'train': s3_input_train,
            'validation': s3_input_validation
        },
        job_name=job_name
    )

    config['model_data'] = xgb.model_data

    with open('config.yaml', 'w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
