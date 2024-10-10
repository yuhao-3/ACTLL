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


nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3_BMM_random\
    --ni 0.1 \
    --label_noise 0 \
    --model ACTLLv3 \
    --modelloss CrossEntropy \
    --batch_size 128\
    --epochs 300 \
    --correct_start 200 \
    --lr 1e-3 \
    --arg_interval 1 \
    --mean_loss_len 10 \
    --gamma 0.3\
    --sel_method 5 \
    --AEChoice TimeAtteCNN\
    --augment True \
    --hard True\
    --corr True\
    --warmup 30\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\


# nohup python src/main.py \
#     --dataset All\
#     --outfile ACTLL_TimeAtteCNNv3_BMM\
#     --ni 0.6 \
#     --label_noise 0 \
#     --model ACTLLv3 \
#     --modelloss CrossEntropy \
#     --batch_size 128\
#     --epochs 300 \
#     --correct_start 200 \
#     --lr 1e-3 \
#     --arg_interval 1 \
#     --mean_loss_len 10 \
#     --gamma 0.3\
#     --sel_method 5 \
#     --AEChoice TimeAtteCNN\
#     --augment True \
#     --hard True\
#     --corr True\
#     --warmup 30\
#     --L_aug_coef 1 \
#     --L_rec_coef 1 \
#     --L_p_coef 0.1\


# nohup python src/main.py \
#     --dataset All\
#     --outfile ACTLL_TimeAtteCNNv3\
#     --ni 0.3 \
#     --label_noise 1 \
#     --model ACTLLv3 \
#     --modelloss CrossEntropy \
#     --batch_size 128\
#     --epochs 300 \
#     --correct_start 200 \
#     --lr 1e-3 \
#     --arg_interval 1 \
#     --mean_loss_len 10 \
#     --gamma 0.3\
#     --sel_method 5 \
#     --AEChoice TimeAtteCNN\
#     --augment True \
#     --hard True\
#     --corr True\
#     --warmup 30\
#     --L_aug_coef 1 \
#     --L_rec_coef 1 \
#     --L_p_coef 0.1\

# nohup python src/main.py \
#     --dataset All\
#     --outfile ACTLL_TimeAtteCNNv3_BMM\
#     --ni 0.4 \
#     --label_noise 1 \
#     --model ACTLLv3 \
#     --modelloss CrossEntropy \
#     --batch_size 128\
#     --epochs 300 \
#     --correct_start 200 \
#     --lr 1e-3 \
#     --arg_interval 1 \
#     --mean_loss_len 10 \
#     --gamma 0.3\
#     --sel_method 1 \
#     --AEChoice TimeAtteCNN\
#     --augment True \
#     --hard True\
#     --corr True\
#     --warmup 30\
#     --L_aug_coef 1 \
#     --L_rec_coef 1 \
#     --L_p_coef 0.1\


# nohup python src/main.py \
#     --dataset All\
#     --outfile ACTLL_TimeAtteCNNv3_corr250\
#     --ni 0.5 \
#     --label_noise 0 \
#     --model ACTLLv3 \
#     --modelloss CrossEntropy \
#     --batch_size 128\
#     --epochs 300 \
#     --correct_start 250 \
#     --lr 1e-3 \
#     --arg_interval 1 \
#     --mean_loss_len 10 \
#     --gamma 0.3\
#     --sel_method 5\
#     --AEChoice TimeAtteCNN\
#     --augment True \
#     --hard True\
#     --corr True\
#     --warmup 30\
#     --L_aug_coef 1 \
#     --L_rec_coef 1 \
#     --L_p_coef 0.1\
    
# nohup python src/main.py \
#     --dataset eICU\
#     --outfile ACTLL_TimeAtteCNNv3_noCorr\
#     --ni 0.5 \
#     --label_noise 0 \
#     --model ACTLLv3 \
#     --modelloss CrossEntropy \
#     --batch_size 512\
#     --epochs 300 \
#     --correct_start 200 \
#     --lr 1e-3 \
#     --arg_interval 1 \
#     --mean_loss_len 10 \
#     --gamma 0.3\
#     --sel_method 5 \
#     --AEChoice TimeAtteCNN\
#     --augment True \
#     --hard False\
#     --corr False\
#     --warmup 30\
#     --L_aug_coef 1 \
#     --L_rec_coef 1 \
#     --L_p_coef 0.1\

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s


