#!/usr/bin/env bash



# Exp Setting
export config_path=configs/svhn
export dataset_type=SVHN
export poison_dataset_type=PoisonSVHN
export attack_type=min-min
export perturb_type=classwise
export base_version=resnet18
export epsilon=8
export step_size=0.8
export num_steps=1
export universal_stop_error=0.01
export universal_train_target='train_subset'
export exp_args=${dataset_type}-eps=${epsilon}-se=${universal_stop_error}-base_version=${base_version}
export exp_path=experiments/svhn/${attack_type}_${perturb_type}/${exp_args}
export scripts_path=scripts/svhn/${attack_type}-noise/${perturb_type}-noise
