#!/bin/bash

# Load Exp Settings
source exp_setting.sh


# Remove previous files
exp_name=${exp_path}/search_noise
echo $exp_name

# Search Universal Perturbation and build datasets
cd ../../../
# rm -rf $exp_name
pwd
python3 perturbation.py --config_path             $config_path       \
                        --exp_name                $exp_name          \
                        --version                 $base_version      \
                        --train_data_type         WebFace            \
                        --test_data_type          WebFace            \
                        --train_data_path         /home/lemonbear/DriveN/data/face-search      \
                        --test_data_path          /home/lemonbear/DriveN/data/face-search      \
                        --noise_shape             150 3 112 112      \
                        --epsilon                 $epsilon           \
                        --num_steps               $num_steps         \
                        --step_size               $step_size         \
                        --attack_type             $attack_type       \
                        --perturb_type            $perturb_type      \
                        --train_step              $train_step        \
                        --train_batch_size        32                 \
                        --eval_batch_size         32                 \
                        --universal_train_target  $universal_train_target\
                        --universal_stop_error    $universal_stop_error\
