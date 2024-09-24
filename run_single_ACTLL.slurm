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
    --dataset Benchmark\
    --outfile ACTLL_TimeCNN_BMM_batch256_noCORRCLUS\
    --ni 0.1 \
    --label_noise 0 \
    --model ACTLL \
    --modelloss CrossEntropy \
    --batch_size 256\
    --epochs 300 \
    --correct_start 10 \
    --correct_end 300 \
    --lr 2e-4 \
    --sel_method 5 \
    --AEChoice TimeAtteCNN\
    --augment True \
    --corr False\


##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage


my-job-stats -a -n -s