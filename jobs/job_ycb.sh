#!/bin/bash
#SBATCH --job-name=download
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --time=2-12:0:0
#SBATCH --output=/kuacc/users/bsefercik/repos/unknown_object_segmentation/jobs/out/job_%j_%x.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bsefercik@ku.edu.tr

#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

/kuacc/users/bsefercik/.conda/envs/py3-mink/bin/python3 /kuacc/users/bsefercik/repos/unknown_object_segmentation/scripts/ycb_generate_point_cloud.py
# sh /kuacc/users/bsefercik/repos/unknown_object_segmentation/scripts/ycb_downloader.sh
