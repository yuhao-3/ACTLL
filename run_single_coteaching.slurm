#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=60G  # Request more memory
#SBATCH --error=slurm-%j.err  # Capture errors in a separate file
#SBATCH --output=slurm-%j.out


# nohup python ./src/SREA_single_experiment.py \
# --dataset Benchmark \
# --epochs 300 \
# --learning_rate 1e-3 \
# --label_noise 0 \
# --M 60 120 180 240 \
# --delta_start 30 \
# --delta_end 90 \
# --embedding_size 32 \
# --ni 0.1 \
# --outfile SREA\

chmod +x run_SOTA_vanilla.sh
./run_SOTA_vanilla.sh



##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s