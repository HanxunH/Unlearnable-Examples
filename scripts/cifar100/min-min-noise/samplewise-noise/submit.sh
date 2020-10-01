#!/bin/bash

# Load Exp Settings
source exp_setting.sh


# Target Models
declare -a type_arr=(
    "resnet18"
    "resnet50"
    "dense121"
    # "resnet18_augmentation"
    # "resnet18_denoise"
)

# Poison Rates
declare -a poison_rate_arr=(
    1.0
    # 0.8
    # 0.6
    # 0.4
    # 0.2
    0.0
)


# Submit Jobs
for model_name in "${type_arr[@]}"
do
    for poison_rate in "${poison_rate_arr[@]}"
    do
      job_name=$exp_args-${model_name}-${poison_rate}
      echo $job_name
      sbatch --partition gpgpu --gres=gpu:1 --time 8:00:00 --job-name $job_name train.slurm $model_name $poison_rate $scripts_path
    done
done


# # Submit Adv Training
# for poison_rate in "${poison_rate_arr[@]}"
#   do
#     job_name=${attack_type}-${perturb_type}-$exp_args-resnet18_madrys-${poison_rate}
#     echo $job_name
#     sbatch --partition gpgpu --gres=gpu:1 --time 12:00:00 --job-name $job_name train.slurm resnet18_madrys $poison_rate $scripts_path
# done


# echo resnet18-madrys-1.0-${exp_args}
# sbatch --partition gpgpu --gres=gpu:1 --time 24:00:00 --job-name ${exp_args}-resnet18-madrys-1.0 train.slurm resnet18_madrys 1.0 $scripts_path
# echo resnet18-madrys-0.0-${exp_args}
# sbatch --partition gpgpu --gres=gpu:1 --time 24:00:00 --job-name ${exp_args}-resnet18-madrys-0.0 train.slurm resnet18_madrys 0.0 $scripts_path
