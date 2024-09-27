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

# chmod +x run_ablation_ACTLL_TimeCNN_BMM.sh
# ./run_ablation_ACTLL_TimeCNN_BMM.sh


python src/main.py \
    --dataset Medical\
    --outfile ACTLL_TimeAtteCNNv3_batch256_corrstart200_schedu_noSpectREAL\
    --ni 0.4 \
    --label_noise 0 \
    --model ACTLLv3 \
    --modelloss CrossEntropy \
    --batch_size 256\
    --epochs 300 \
    --correct_start 200 \
    --lr 2e-3 \
    --sel_method 5 \
    --AEChoice TimeAtteCNN\
    --augment True \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\


##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage


my-job-stats -a -n -s