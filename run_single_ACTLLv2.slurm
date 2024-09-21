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

python src/main.py \
    --dataset Benchmark \
    --outfile ACTLL_TimeCNN_BMMv2_CrossEntorpy_newCoef_batch256_cosine \
    --ni 0.1 \
    --label_noise 0 \
    --model ACTLLv2 \
    --modelloss CrossEntropy \
    --epochs 300 \
    --batch_size 256 \
    --lr 2e-4 \
    --sel_method 5 \
    --AEChoice TimeAtteCNN \
    --augment True \
    --corr True \
    --correct_start 10 \
    --correct_end 300 \
    --mean_loss_len 10 \
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_cls_coef 0.1 \
    --L_cor_coef 0.1 \
    --L_p_coef 0.1 \
    --L_e_coef 0.1



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s