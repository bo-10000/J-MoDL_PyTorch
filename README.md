# MoDL

PyTorch implementation of  J-MoDL: Joint Model-Based Deep Learning for Optimized Sampling and Reconstruction (Not official!)

Official code: https://github.com/hkaggarwal/J-MoDL

![alt text](https://github.com/hkaggarwal/J-MoDL/blob/master/j-modl_architecture.jpg)

## Reference paper

J-MoDL: Joint Model-Based Deep Learning for Optimized Sampling and Reconstruction by H.K Aggarwal and M. Jacob in IEEE Journal of Selected Topics in Signal Processing, (2020).

Link: https://arxiv.org/abs/1911.02945

IEEE Xplore: https://ieeexplore.ieee.org/document/9122388

## Dataset

Subset of the multi-coil brain dataset used in the original paper is publically available. Test dataset `tstdata_jmodl.npz` and initial mask `initmask6.npz` are already included in the `data` folder. Please download the train dataset from the following link and locate in under the `data` directory.

**Download Link** : https://drive.google.com/file/d/1GLqs2A5YpRN8RdDJgdhrspL3zjlG0Qha/view?usp=sharing

## Configuration file

The configuration files are in `config` folder. Every setting is the same as the authors used in their official repo, but not the same as the ones used in the paper.

## Train

You can change the configuration file for training by modifying the `train.sh` file.

```
scripts/train.sh
```

## Test

You can change the configuration file for testing by modifying the `test.sh` file.

```
scripts/test.sh
```

## Saved models

Saved models are provided.

`workspace/base_modl/checkpoints/final.epoch0099-score38.9911.pth` 
