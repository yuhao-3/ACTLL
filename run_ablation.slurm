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

chmod +x run_ablation_ACTLL_TimeCNN_GMM.sh
./run_ablation_ACTLL_TimeCNN_GMM.sh

chmod +x run_ablation_ACTLL_TimeCNN_Sloss.sh
./run_ablation_ACTLL_TimeCNN_Sloss.sh

# # # ############ Encoder Architecture

chmod +x run_ablation_ACTLL_CNN_BMM.sh
./run_ablation_ACTLL_CNN_BMM.sh

chmod +x run_ablation_ACTLL_Diffusion_BMM.sh
./run_ablation_ACTLL_Diffusion_BMM.sh

# # # ############## Correction 

chmod +x run_ablation_ACTLL_TimeCNN_BMM_nocorr.sh
./run_ablation_ACTLL_TimeCNN_BMM_nocorr.sh

# # # ############## Augmentation

chmod +x run_ablation_ACTLL_TimeCNN_BMM_noaug.sh
./run_ablation_ACTLL_TimeCNN_BMM_noaug.sh


##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s