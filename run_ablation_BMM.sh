#!/bin/bash

# add '--ucr 128' for training all datasets in UCR
# --label_noise 0: symmetric noise
# --label_noise 1: asymmetric noise
# --label_noise -1: instance-depended noise

############# Symmetric NOISE #############
nohup python src/BMM_single_experiment.py \
    --dataset MIMIC \
    --outfile MixUp_BMM \
    --ni 0.1 \
    --label_noise 0 \
    --epoch 100 \
    --lr 1e-4 \

nohup python src/BMM_single_experiment.py \
    --dataset MIMIC \
    --outfile MixUp_BMM \
    --ni 0.2 \
    --label_noise 0 \
    --epoch 100 \
    --lr 1e-4 \


nohup python src/BMM_single_experiment.py \
    --dataset MIMIC \
    --outfile MixUp_BMM \
    --ni 0.3 \
    --label_noise 0 \
    --epoch 100 \
    --lr 1e-4 \


nohup python src/BMM_single_experiment.py \
    --dataset MIMIC \
    --outfile MixUp_BMM\
    --ni 0.4 \
    --label_noise 0 \
    --epoch 100 \
    --lr 1e-4 \


nohup python src/BMM_single_experiment.py \
    --dataset MIMIC \
    --outfile MixUp_BMM \
    --ni 0.5 \
    --label_noise 0 \
    --epoch 100 \
    --lr 1e-4 \



# ######## Asymmetric NOISE #############
# nohup python src/BMM_single_experiment.py \
#     --dataset MIMIC \
#     --outfile MixUp_BMM \
#     --ni 0.1 \
#     --label_noise 1 \
#     --epoch 100 \
#     --lr 1e-4 \

# nohup python src/BMM_single_experiment.py \
#     --dataset MIMIC \
#     --outfile MixUp_BMM\
#     --ni 0.2 \
#     --label_noise 1 \
#     --epoch 100 \
#     --lr 1e-4 \


# nohup python src/BMM_single_experiment.py \
#     --dataset MIMIC \
#     --outfile MixUp_BMM \
#     --ni 0.3 \
#     --label_noise 1 \
#     --epoch 100 \
#     --lr 1e-4 \


# ######## INSTANCE DEPENDENT NOISE #############

# nohup python src/BMM_single_experiment.py \
#     --dataset MIMIC \
#     --outfile MixUp_BMM\
#     --ni 0.3 \
#     --label_noise -1 \
#     --epoch 100 \
#     --lr 1e-4 \


# nohup python src/BMM_single_experiment.py \
#     --dataset MIMIC \
#     --outfile MixUp_BMM \
#     --ni 0.4 \
#     --label_noise -1 \
#     --epoch 100 \
#     --lr 1e-4 \
# ###########################################################################################################


