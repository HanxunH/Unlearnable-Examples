#!/usr/bin/env bash



# Exp Setting
export config_path=configs/tiny-imagenet
export dataset_path=../datasets/ILSVRC2012
export dataset_type=TinyImageNet
export poison_dataset_type=PoisonImageNetMini
export attack_type=min-min
export perturb_type=classwise
export base_version=resnet18
export epsilon=16
export step_size=1.6
export num_steps=1
export train_step=250
export universal_stop_error=0.1
export universal_train_target='train_subset'
export exp_args=${dataset_type}-eps=${epsilon}-se=${universal_stop_error}-base_version=${base_version}
export exp_path=experiments/tiny-imagenet/${attack_type}_${perturb_type}/${exp_args}
export scripts_path=scripts/tiny-imagenet/${attack_type}-noise/${perturb_type}-noise
