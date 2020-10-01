#!/bin/bash

# Load EXP Setting
source exp_setting.sh

# Training Setting
model_name=$1
poison_rate=$2
exp_name=${exp_path}
echo $exp_name

# Poison Training
cd ../../../
rm -rf ${exp_name}/${model_name}
python3 -u main.py    --version                 $model_name                 \
                      --exp_name                $exp_name                   \
                      --config_path             $config_path                \
                      --train_data_path         $dataset_path               \
                      --test_data_path          $test_dataset_path          \
                      --train_data_type         $dataset_type               \
                      --test_data_type          $dataset_type               \
                      --train_batch_size        64                         \
                      --eval_batch_size         64                         \
                      --train --data_parallel --train_face
