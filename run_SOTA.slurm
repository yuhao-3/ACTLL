#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# # Run each script concurrently in the background
# chmod +x scripts/run_SOTA_SREA.sh
# ./scripts/run_SOTA_SREA.sh 

# chmod +x scripts/run_SOTA_coteaching.sh
# ./scripts/run_SOTA_coteaching.sh 

# chmod +x scripts/run_SOTA_vanilla.sh
# ./scripts/run_SOTA_vanilla.sh 

# chmod +x scripts/run_SOTA_dividemix.sh
# ./scripts/run_SOTA_dividemix.sh 

# chmod +x scripts/run_SOTA_SIGUA.sh
# ./scripts/run_SOTA_SIGUA.sh 



# eICU

# chmod +x scripts/run_SOTA_ACTLL_TimeCNN_BMM_all.sh
# ./scripts/run_SOTA_ACTLL_TimeCNN_BMM_all.sh 

# chmod +x scripts/run_SOTA_BMM.sh
# ./scripts/run_SOTA_BMM.sh 

# chmod +x scripts/run_SOTA_CTW.sh
# ./scripts/run_SOTA_CTW.sh 



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s