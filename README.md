# <img src= "picture/unimelb.png"  width=20% align=left>  Attention-based learning with dynamic Calibration and augmentation for Time series noisy Label Learning 


## Abstract
 Medical research, including predicting patient outcomes, heavily relies on medical time series data collected through Electronic Health Records(EHR) with rich information on patient history. Despite careful examination of medical data, labeling errors originating from human annotation are an unavoidable phenomenon, posing significant disturbance to the effective prediction of patient outcomes. Thus, we propose a learning of Attention-based learning with dynamic Calibration and augmentation for Time series noisy Label Learning that captures better temporal dynamics, as well as dynamically calibrates uncertain class labels or augments confident instances based on mean of two components by each mixture models per class. Experiments under eICU, MIMIC-IV-ED and several benchmark datasets from UCR and UEA repository have illustrated comparable or even state-of-the-art results. Our code is available at: https://github.com/yuhao-3/ACTLL.


<div align="center">
<img src="picture/Model.png" width="70%">
</div>

## Data
### Benchmark Data
We evaluate our model on publicly available time-series classification datasets from the UCR and UEA repositories:

[The UCR time series archive](https://ieeexplore.ieee.org/abstract/document/8894743)

[The UEA multivariate time series classification archive, 2018](https://arxiv.org/abs/1811.00075)

- Note: you may have to manually download: Univariate2018_ts.zip and Multivariate2018_arff.zip
- Then decompress and rename the decompressed folder as  'ucr\_data' and 'uea\_data' respectively
- Put them into 'src/ucr_data', 'src/uea_data' directory

### EHR data
For the EHR datasets, you need to get download access first before proceeding:

[MIMIC-IV-ED](https:/physionet.org/content/mimiciv/1.0/)

To get preprocessed MIMIC-IV-ED dataset, follow the same preprocessing procedure as [CAMELOT](https://github.com/hrna-ox/camelot-icml)
- Download the MIMIC-IV-ED
Download the core directory from MIMIC-IV
- Download the data preprocessing pipeline: [link](https://github.com/hrna-ox/camelot-icml/tree/main/src/data_processing/MIMIC)
- Run the scripts by python run_processing.py 
- Put folder "MIMIC" into 'src/EHR\_data/MIMIC'

[eICU](https://physionet.org/content/eicu-crd/2.0/.)

To get preprocessed eICU dataset: Follow the same procedure in eICU Pre-Processing section in [GNN-LSTM](https://github.com/EmmaRocheteau/eICU-GNN-LSTM)
- Note: eICU dataset can be extremely cumbersome to process, please be patient.
- Put folder "eICU" into 'src/EHR\_data/eICU'


## Requirements
The packages our code depends on are in ```./requirements.txt```.

## Usage
To train ACTLL on 15 benchmark datasets mentioned in this paper, run
```bash
nohup python ./src/main.py --model ACTLLv3 --epochs 300 --lr 1e-3 --label_noise 0 --ni 0.3 --num_workers 1 --mean_loss_len 10 --gamma 0.3 --cuda_device 0 --outfile CTW.csv
```
The results are put in ```./statistic_results/```. 

For other examples, please refer to ```./run_sym30.sh```.


## Acknowledgement
We adapted the following open-source code to implement the state-of-the-art algorithms
* [SIGUA](https://github.com/bhanML/SIGUA)
* [Co-teaching](https://github.com/bhanML/Co-teaching) 
* [MixUp-BMM](https://github.com/PaulAlbert31/LabelNoiseCorrection)
* [Dividemix](https://github.com/LiJunnan1992/DivideMix)
* [SREA](https://github.com/Castel44/SREA)
* [CTW](https://github.com/qianlima-lab/CTW)