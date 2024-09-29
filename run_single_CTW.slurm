#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=60G  # Request more memory
#SBATCH --error=slurm-%j.err  # Capture errors in a separate file
#SBATCH --output=slurm-%j.out



# Navigate to the directory containing run_processing.py


# Run the Python script



python src/main.py \
    --dataset eICU\
    --outfile CTW_dummy\
    --ni 0.4 \
    --label_noise -1 \
    --model CTW \
    --epochs 300 \
    --batch_size 512 \
    --lr 1e-3 \
    --arg_interval 1 \
    --mean_loss_len 10 \
    --gamma 0.3\

# chmod +x run_SOTA_CTW.sh
# ./run_SOTA_CTW.sh


##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s