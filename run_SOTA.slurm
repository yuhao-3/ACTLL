#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=60G  # Request more memory
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err  # Capture errors in a separate file



# chmod +x run_SOTA_ACTLL_TimeCNN_BMM_all.sh
# ./run_SOTA_ACTLL_TimeCNN_BMM_all.sh

# chmod +x run_SOTA_CTW.sh
# ./run_SOTA_CTW.sh

# chmod +x run_SOTA_SREA.sh
# ./run_SOTA_SREA.sh

# chmod +x run_SOTA_BMM.sh
# ./run_SOTA_BMM.sh

chmod +x run_SOTA_dividemix.sh
./run_SOTA_dividemix.sh

chmod +x run_SOTA_coteaching.sh
./run_SOTA_coteaching.sh

chmod +x run_SOTA_SIGUA.sh
./run_SOTA_sigua.sh

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s