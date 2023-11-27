# VCD
Code for [Variational Causal Dynamics](https://openreview.net/forum?id=V9tQKYYNK1)

This repository contains the model, training scripts and data generation code for VCD. 

## Install
1. Clone this repository
2. Install dependencies using conda
```
conda env create -f env.yml
```

## Run experiments
To reproduce the results in the paper, use the jupyter notebooks provided in the /experiments/ folder. Pretrained models are available in the /pretrain/ folde.

## Train models
To train models from scratch, use the scripts in the /training/ folder. Note that the code should be run in the /training/ folder. The model hyperparameters can be easily modified in /models/xxx.json. 

Before training the models, please fill in the wandb credentials in the training scripts.

For example, to train the mixed-state VCD, edit /training/train_mixed_state.py to fill in wandb credentials, then run the following:
```
cd training
python train_mixed_state.py --model_config ../models/mixed_state_VCD_conf.json
```
For image experiments, it is recommended to initialise the model with a VAE pretrained purely on image reconstruction. The pretrained weights are provided in /pretrain/pretrain_vae.npy. The training script is also available in /training/pretrain_vae.py.

To initialise the model with the pretrained VAE weights, run:
```
cd training
python train_image.py --model_config ../models/image_VCD_conf.json --pretrain_path ../pretrain/pretrain_vae.npy
```

## Cite

The paper is available [here](https://openreview.net/forum?id=V9tQKYYNK1), and can be cited with the following bibtex entry:

```
@article{
lei2023variational,
title={Variational Causal Dynamics: Discovering Modular World Models from Interventions},
author={Anson Lei and Bernhard Schölkopf and Ingmar Posner},
journal={Transactions on Machine Learning Research},
year={2023},
url={https://openreview.net/forum?id=V9tQKYYNK1}
}
```