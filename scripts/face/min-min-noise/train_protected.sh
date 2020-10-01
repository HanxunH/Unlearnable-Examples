#!/bin/bash

# Load EXP Setting
source exp_setting.sh

# Training Setting
model_name=$1
exp_name=${exp_path}/protected_train
echo $exp_name

cd ../../../
rm -rf ${exp_name}/${model_name}
python3 -u main.py    --version                 $model_name                 \
                      --exp_name                $exp_name                   \
                      --config_path             $config_path                \
                      --train_data_path         ../datasets/casia-112x112-protected \
                      --test_data_path          $test_dataset_path          \
                      --train_data_type         $dataset_type               \
                      --test_data_type          $dataset_type               \
                      --train_batch_size        512                         \
                      --eval_batch_size         512                         \
                      --train --train_face --data_parallel
