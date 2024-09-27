#!/bin/bash

# add '--ucr 128' for training all datasets in UCR
# --label_noise 0: symmetric noise
# --label_noise 1: asymmetric noise
# --label_noise -1: instance-depended noise

############# Symmetric NOISE #############
nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise 0 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.1 \
    --outfile SREA\

nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise 0 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.2 \
    --outfile SREA\


nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise 0 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.3 \
    --outfile SREA\


nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise 0 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.4 \
    --outfile SREA\


nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise 0 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.5 \
    --outfile SREA\





######## Asymmetric NOISE #############
nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise 1 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.1 \
    --outfile SREA\

nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise 1 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.2 \
    --outfile SREA\


nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise 1 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.3 \
    --outfile SREA\


######## INSTANCE DEPENDENT NOISE #############

nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise -1 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.3 \
    --outfile SREA\


nohup python ./src/SREA_single_experiment.py \
    --dataset All \
    --epochs 300 \
    --learning_rate 1e-3 \
    --label_noise -1 \
    --M 60 120 180 240 \
    --delta_start 30 \
    --delta_end 90 \
    --embedding_size 32 \
    --ni 0.4 \
    --outfile SREA\
###########################################################################################################


