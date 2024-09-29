#!/bin/bash

# add '--ucr 128' for training all datasets in UCR
# --label_noise 0: symmetric noise
# --label_noise 1: asymmetric noise
# --label_noise -1: instance-depended noise

############# Symmetric NOISE #############
nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_CNNv3\
    --ni 0.3 \
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
    --AEChoice CNN\
    --augment True \
    --hard True\
    --corr True\
    --warmup 30\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\


nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_Diffusionv3\
    --ni 0.3 \
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
    --AEChoice Diffusion\
    --augment True \
    --hard True\
    --corr True\
    --warmup 30\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\