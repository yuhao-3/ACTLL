#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=60G  # Request more memory
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err  # Capture errors in a separate file


# Navigate to the directory containing run_processing.py


# ############### Sample selection method
chmod +x run_ablation_ACTLL_TimeCNN_SampleSel.sh
./run_ablation_ACTLL_TimeCNN_SampleSel.sh

# # # ############ Encoder Architecture
chmod +x run_ablation_ACTLL_encoder.sh
./run_ablation_ACTLL_encoder.sh

# # # ############## Module analysis 
chmod +x run_ablation_ACTLL_TimeCNN_module.sh
./run_ablation_ACTLL_TimeCNN_module.sh



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s