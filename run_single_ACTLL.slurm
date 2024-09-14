#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=60G  # Request more memory
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err  # Capture errors in a separate file




# Navigate to the directory containing run_processing.py


# Run the Python script

chmod +x run_ablation_ACTLL_TimeCNN_BMM.sh
./run_ablation_ACTLL_TimeCNN_BMM.sh


##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s