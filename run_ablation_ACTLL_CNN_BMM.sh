#!/bin/bash

# add '--ucr 128' for training all datasets in UCR
# --label_noise 0: symmetric noise
# --label_noise 1: asymmetric noise
# --label_noise -1: instance-depended noise

############# Symmetric NOISE #############
nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.1 \
    --label_noise 0 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN \
    --augment True \
    --corr True \

nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.2 \
    --label_noise 0 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN\
    --augment True \
    --corr True \


nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.3 \
    --label_noise 0 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN\
    --augment True \
    --corr True \


nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.4 \
    --label_noise 0 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN\
    --augment True \
    --corr True \


nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.5 \
    --label_noise 0 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN\
    --augment True \
    --corr True \



######## Asymmetric NOISE #############


nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.1 \
    --label_noise 1 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN\
    --augment True \
    --corr True \

nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.2 \
    --label_noise 1 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN\
    --augment True \
    --corr True \


nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.3 \
    --label_noise 1 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN\
    --augment True \
    --corr True \


######## INSTANCE DEPENDENT NOISE #############

nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.3 \
    --label_noise -1 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN \
    --augment True \
    --corr True \

nohup python src/main.py \
    --dataset Medical \
    --outfile ACTLL_CNN_BMM \
    --ni 0.4 \
    --label_noise -1 \
    --model ACTLL \
    --modelloss Focal \
    --epochs 100 \
    --sel_method 5 \
    --AEChoice CNN \
    --augment True \
    --corr True \
###########################################################################################################


