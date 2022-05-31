#!/bin/bash
#SBATCH --job-name=kp2pose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --time=4-12:0:0
#SBATCH --output=/kuacc/users/bsefercik/repos/unknown_object_segmentation/jobs/out/job_%j_%x.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bsefercik@ku.edu.tr

#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --constraint=ai
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

module load cuda/11.1
module load gcc/8.3.0
export CUDA_HOME=/usr/local/cuda-11.1
export OMP_NUM_THREADS=6

/kuacc/users/bsefercik/.conda/envs/py3-mink-v100/bin/python3 \
/kuacc/users/bsefercik/repos/unknown_object_segmentation/train_kp_to_pose.py \
--config /kuacc/users/bsefercik/repos/unknown_object_segmentation/config/default.yaml \
--override /kuacc/users/bsefercik/repos/unknown_object_segmentation/exp/exp46_kp2pose/default.yaml