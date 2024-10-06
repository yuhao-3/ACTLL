#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=60G  # Request more memory
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err  # Capture errors in a separate file


############## Module analysis 
chmod +x scripts/run_ablation_ACTLL_TimeCNN_module.sh
./scripts/run_ablation_ACTLL_TimeCNN_module.sh  # Run in the background


############## Sample selection method
chmod +x scripts/run_ablation_ACTLL_TimeCNN_SampleSel.sh
./scripts/run_ablation_ACTLL_TimeCNN_SampleSel.sh   # Run in the background

########### Encoder Architecture
chmod +x scripts/run_ablation_ACTLL_encoder.sh
./scripts/run_ablation_ACTLL_encoder.sh  # Run in the background

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s