#!/bin/bash

# add '--ucr 128' for training eICU datasets in UCR
# --label_noise 0: symmetric noise
# --label_noise 1: asymmetric noise
# --label_noise -1: instance-depended noise

# ############# Symmetric NOISE #############
nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU \
    --ni 0.1 \
    --label_noise 0 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \

nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU \
    --ni 0.2 \
    --label_noise 0 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \


nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU \
    --ni 0.3 \
    --label_noise 0 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \


nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU\
    --ni 0.4 \
    --label_noise 0 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \


nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU \
    --ni 0.5 \
    --label_noise 0 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \



# ######## Asymmetric NOISE #############
nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU \
    --ni 0.1 \
    --label_noise 1 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \

nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU\
    --ni 0.2 \
    --label_noise 1 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \


nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU \
    --ni 0.3 \
    --label_noise 1 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \


# # ######## INSTANCE DEPENDENT NOISE #############

nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU\
    --ni 0.3 \
    --label_noise -1 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \


nohup python src/BMM_single_experiment.py \
    --dataset eICU \
    --outfile MixUp_BMM_eICU \
    --ni 0.4 \
    --label_noise -1 \
    --batch_size 512\
    --epoch 300 \
    --lr 1e-3 \
# ###########################################################################################################


