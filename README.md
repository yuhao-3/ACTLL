# <img src= "picture/unimelb.png"  width=20% align=left>  ACTLL: Adaptive confidence of label calibration for label noise learning



This is the training code for our work "CTW: Confident Time-Warping for Time-Series Label-Noise Learning". 

## Abstract


<div align="center">
<img src="picture/Model.png" width="70%">
</div>

## Data
We evaluate our model on publicly available time-series classification datasets from the UCR and UEA repositories:

[The UCR time series archive](https://ieeexplore.ieee.org/abstract/document/8894743)

[The UEA multivariate time series classification archive, 2018](https://arxiv.org/abs/1811.00075)

All datasets are downloaded automatically during training if they do not exist.

## Requirements
The packages our code depends on are in ```./requirements.txt```.

## Usage
To train CTW on 13 benchmark datasets mentioned in this paper, run
```bash
nohup python ./src/main.py --model CTW --epochs 300 --lr 1e-3 --label_noise 0 --embedding_size 32 --ni 0.3 --num_workers 1 --mean_loss_len 10 --gamma 0.3 --cuda_device 0 --outfile CTW.csv >/dev/null 2>&1 &
```
The results are put in ```./statistic_results/```. 
(P.S. The evaluation process of CTW is at *line 231* in ./src/utils/training_helper_CTW.py )

For other examples, please refer to ```./run_sym30.sh```.


## Acknowledgement
We adapted the following open-source code to implement the state-of-the-art algorithms
* [SIGUA](https://github.com/bhanML/SIGUA)
* [Co-teaching](https://github.com/bhanML/Co-teaching) 
* [MixUp and MixUp-GMM](https://github.com/PaulAlbert31/LabelNoiseCorrection)
* [SREA](https://github.com/Castel44/SREA)
* [Dividemix](https://github.com/LiJunnan1992/DivideMix)
* [Sel-CL](https://github.com/ShikunLi/Sel-CL)
* [CTW](https://github.com/qianlima-lab/CTW)