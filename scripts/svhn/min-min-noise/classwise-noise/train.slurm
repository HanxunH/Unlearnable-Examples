#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name="c100-universal"
#SBATCH --output=slurm-%A-%x.out
#SBATCH --account="punim0784"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time 4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pineappleorcas@gmail.com

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Training Setting
model_name=$1
poison_rate=$2
scripts_path=$3

# Run the job from this directory:
cd /data/gpfs/projects/punim0784/min-min-noise
cd $scripts_path

# Load EXP Setting
source exp_setting.sh

exp_name=${exp_path}/poison_train_${poison_rate}
echo $exp_name

# Poison Training
cd ../../../../
rm -rf ${exp_name}/${model_name}
python3 -u main.py    --version                 $model_name                 \
                      --exp_name                $exp_name                   \
                      --config_path             $config_path                \
                      --train_data_type         $poison_dataset_type        \
                      --test_data_type          $dataset_type               \
                      --poison_rate             $poison_rate                \
                      --perturb_type            $perturb_type               \
                      --perturb_tensor_filepath ${exp_path}/perturbation.pt \
                      --train
