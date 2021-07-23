# Batch Augmentation Implementation with Pytorch
- Unofficial implementation of the paper *Augment Your Batch: Improving Generalization Through Instance Repetition*


## 0. Develop Environment
```
Docker Image
- pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
```
- Using Single GPU


## 1. Implementation Details
- augmentation.py : Cutout implementation
- dataset.py : make repetition of dataset
- model.py : Wide ResNet model
- train.py : train Wide ResNet
- utils.py : count correct prediction
- BA - Wide ResNet 28 10 - Cifar 10 (M=6, Single) : install library, download dataset, preprocessing, train and result
- BA - Wide ResNet 28 10 - Cifar 10 (M=6, 5 Average) : average 5 runs
- Details
  * Follow Cutout train details
    * batch size 128, learning rate 0.1, nesterov momentum 0.9, weight decay 0.0005
    * learning rate schedulers on [60, 120, 160] with 0.2 learning rate drop
    * augmentation : mean/std preprocessing, pad and crop


## 2. Result Comparison on CIFAR-10
|Source|Score|Detail|
|:-:|:-:|:-|
|Paper|97.15|BA with Cutout, WRN-28-10|
|Current Repo|97.568|BA with Cutout, WRN-28-10, Average 5 runs|


## 3. Reference
- Augment Your Batch: Improving Generalization Through Instance Repetition [[paper]](https://arxiv.org/pdf/1901.09335.pdf)
