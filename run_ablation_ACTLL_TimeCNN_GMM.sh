#!/bin/bash

# add '--ucr 128' for training all datasets in UCR
# --label_noise 0: symmetric noise
# --label_noise 1: asymmetric noise
# --label_noise -1: instance-depended noise

############# Symmetric NOISE #############
nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
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
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\

nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
    --ni 0.2 \
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
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\


nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
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
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\


nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
    --ni 0.4 \
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
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\


nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
    --ni 0.5 \
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
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\



######## Asymmetric NOISE #############


nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
    --ni 0.1 \
    --label_noise 1 \
    --model ACTLLv3 \
    --modelloss CrossEntropy \
    --batch_size 128\
    --epochs 300 \
    --correct_start 200 \
    --lr 1e-3 \
    --arg_interval 1 \
    --mean_loss_len 10 \
    --gamma 0.3\
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\

nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
    --ni 0.2 \
    --label_noise 1 \
    --model ACTLLv3 \
    --modelloss CrossEntropy \
    --batch_size 128\
    --epochs 300 \
    --correct_start 200 \
    --lr 1e-3 \
    --arg_interval 2 \
    --mean_loss_len 10 \
    --gamma 0.3\
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\


nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
    --ni 0.3 \
    --label_noise 1 \
    --model ACTLLv3 \
    --modelloss CrossEntropy \
    --batch_size 128\
    --epochs 300 \
    --correct_start 200 \
    --lr 1e-3 \
    --arg_interval 1 \
    --mean_loss_len 10 \
    --gamma 0.3\
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\


######## INSTANCE DEPENDENT NOISE #############

nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
    --ni 0.3 \
    --label_noise -1 \
    --model ACTLLv3 \
    --modelloss CrossEntropy \
    --batch_size 128\
    --epochs 300 \
    --correct_start 200 \
    --lr 1e-3 \
    --arg_interval 1 \
    --mean_loss_len 10 \
    --gamma 0.3\
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\

nohup python src/main.py \
    --dataset All\
    --outfile ACTLL_TimeAtteCNNv3\
    --ni 0.4 \
    --label_noise -1 \
    --model ACTLLv3 \
    --modelloss CrossEntropy \
    --batch_size 128\
    --epochs 300 \
    --correct_start 200 \
    --lr 1e-3 \
    --arg_interval 1 \
    --mean_loss_len 10 \
    --gamma 0.3\
    --sel_method 2 \
    --AEChoice TimeAtteCNN\
    --augment False \
    --hard True\
    --corr True\
    --warmup 20\
    --L_aug_coef 1 \
    --L_rec_coef 1 \
    --L_p_coef 0.1\
###########################################################################################################


