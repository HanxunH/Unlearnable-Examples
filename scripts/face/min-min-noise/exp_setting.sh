#!/usr/bin/env bash
# Exp Setting
export config_path=configs/face
export dataset_path=../datasets/casia-112x112-protected-train
export test_dataset_path=../datasets/casia-112x112-protected-val
export dataset_type=WebFace
export poison_dataset_type=WebFace
export base_version=InceptionResnet
export attack_type=min-min
export perturb_type=classwise
export epsilon=16
export step_size=1.6
export num_steps=1
export train_step=30
export universal_stop_error=0.1
export universal_train_target='train_dataset'
export exp_path=experiments/face
export scripts_path=scripts/face
