#!/bin/bash



# Exp Setting
export config_path=configs/cifar10
export dataset_type=CIFAR10
export poison_dataset_type=PoisonCIFAR10
export attack_type=min-min
export perturb_type=samplewise
export base_version=resnet18
export epsilon=8
export step_size=0.8
export num_steps=20
export universal_stop_error=0.01
export universal_train_target='train_dataset'
export exp_args=${dataset_type}-eps=${epsilon}-se=${universal_stop_error}-base_version=${base_version}
export exp_path=experiments/cifar10/${attack_type}_${perturb_type}/${exp_args}
