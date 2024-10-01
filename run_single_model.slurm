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


# ######## Asymmetric NOISE #############
nohup python src/main.py \
    --dataset eICU \
    --outfile CTW \
    --ni 0.4 \
    --label_noise -1 \
    --batch_size 512\
    --model CTW \
    --epochs 300 \
    --lr 1e-3 \
    --arg_interval 1 \
    --mean_loss_len 10 \
    --gamma 0.3\

###########################################################################################################





# python src/main.py \
#     --dataset eICU \
#     --outfile CTW \
#     --ni 0.1 \
#     --label_noise 0 \
#     --model CTW \
#     --batch_size 512\
#     --epochs 300 \
#     --lr 1e-3 \
#     --arg_interval 1 \
#     --mean_loss_len 10 \
#     --gamma 0.3\



# chmod +x run_SOTA_ACTLL_TimeCNN_BMM_all.sh
# ./run_SOTA_ACTLL_TimeCNN_BMM_all.sh



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s


