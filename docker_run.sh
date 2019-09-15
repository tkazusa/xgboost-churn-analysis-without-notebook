#!/bin/sh
sudo docker run --name sagemaker-xgboost -d -it \
  -v $(pwd):$HOME/work \
  xgboost-container-base:0.90-1-cpu-py3 /bin/bash

