#!/bin/bash

# Load EXP Setting
source exp_setting.sh


# Training Setting
model_name=$1
poison_rate=1.0
exp_name=${exp_path}/poison_train_${poison_rate}
echo $exp_name

# Poison Training
cd ../../
rm -rf ${exp_name}/${model_name}
python3 -u main.py    --version                 $model_name                 \
                      --exp_name                experiments/cifar101_transfer \
                      --config_path             configs/cifar101            \
                      --train_data_type         PoisonCIFAR101              \
                      --test_data_type          PoisonCIFAR101              \
                      --poison_rate             $poison_rate                \
                      --perturb_type            $perturb_type               \
                      --perturb_tensor_filepath ${exp_path}/perturbation.pt \
                      --train
