#!/bin/bash

#SBATCH --account=dl66
#SBATCH --partition=dgx
#SBATCH --qos=dgx

#SBATCH -n 1
#SBATCH -c 40
#SBATCH --gres=gpu:V100:8
#SBATCH --mem=400GB
#SBATCH --time=24:00:00

#SBATCH --mail-user=zizhengpan98@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --exclude=dgx000


# Command to run a gpu job
# For example:
# module load anaconda/2019.03-Python3.7-gcc5
module load gcc/5.4.0
module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
export PROJECT=dl65
export CONDA_ENVS_PATH=/projects/$PROJECT/$USER/conda_envs
export CONDA_PKGS_DIRS=/projects/$PROJECT/$USER/conda_pkgs
source activate /projects/$PROJECT/$USER/conda_envs/mmlab-det
which python

nvidia-smi
cd ../

bash tools/dp_train.sh configs/lit/mask_rcnn_lit_s_patch4_800_adamw_1x_coco.py8




