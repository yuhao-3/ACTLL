#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=60G  # Request more memory
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err  # Capture errors in a separate file


# Navigate to the directory containing run_processing.py


# Run the Python script
# chmod +x run_ablation.sh

### SOTA MODELS

# chmod +x run_ablation_CTW.sh
# ./run_ablation_CTW.sh




# chmod +x run_ablation.sh
# ./run_ablation.sh


# chmod +x run_ablation_ACTLL_TimeCNN_BMM.sh
# ./run_ablation_ACTLL_TimeCNN_BMM.sh

# chmod +x run_ablation_Sel_CL.sh
# ./run_ablation_Sel_CL.sh


chmod +x run_ablation_SREA.sh
./run_ablation_SREA.sh

chmod +x run_ablation_BMM.sh
./run_ablation_BMM.sh


chmod +x run_ablation_dividemix.sh
./run_ablation_dividemix.sh

chmod +x run_ablation_coteaching.sh
./run_ablation_coteaching.sh

# chmod +x run_ablation_ACTLL_Diff_BMM_all.sh
# ./run_ablation_ACTLL_Diff_BMM.sh

# # ############ Sel Method

# chmod +x run_ablation_ACTLL_Diff_GMM.sh
# ./run_ablation_ACTLL_Diff_GMM.sh

# chmod +x run_ablation_ACTLL_Diff_noSel.sh
# ./run_ablation_ACTLL_Diff_noSel.sh

# # # ############ Encoder Architecture

# chmod +x run_ablation_ACTLL_CNN_BMM.sh
# ./run_ablation_ACTLL_CNN_BMM.sh

# chmod +x run_ablation_ACTLL_AtteDiff_BMM.sh
# ./run_ablation_ACTLL_AtteDiff_BMM.sh

# # ############## Correction 

# chmod +x run_ablation_ACTLL_TimeCNN_BMM_nocorr.sh
# ./run_ablation_ACTLL_TimeCNN_BMM_nocorr.sh

# # ############## Augmentation

# chmod +x run_ablation_ACTLL_TimeCNN_BMM_noaug.sh
# ./run_ablation_ACTLL_TimeCNN_BMM_noaug.sh


##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s