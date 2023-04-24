[README](./README.md)
====

Official repository for *Interpreting Vulnerabilities of Multi-Instance Learning to Adversarial Perturbations*.<br>
Any question can contact with inki.yinji@gmail.com<br>
My home pages:
  * **Blog**: [https://inkiyinji.blog.csdn.net](https://inkiyinji.blog.csdn.net "Inki's blog")
  * **Data**： [https://www.kaggle.com/inkiyinji](https://www.kaggle.com/inkiyinji "Inki's kaggle")

****

## Introduction

The code implements two MIL attackers: MI-CAP and MI-UAP

If the input is a bag with images, we annotate the file names as MI-CAP2D and MI-UAP2D<br>
If the input is a bag with vectors, we annotate the file names as MI-CAP and MI-UAP

## How to use

For ShanghaiTech and UCF-Crime data sets, just run MI-CAP and MI-UAP<br>
For MNIST, CIFAR10, and STL10, just run MI-CAP2D and MI-UAP2D<br>

For experimental parameters:
  * xi: The magnitude of perturbation
    * For ShanghaiTech and UCF-crime: the default setting is 0.01
    * For images: 0.2
  * mode: The computation mode for gradient, "ave" or "att"
  * net_type: The attacked network, the choice includes:
    1. ab: ABMIL
    2. ga: GAMIL
    3. la: LAMIL
    4. ds: DSMIL
    5. ma: MAMIL
  * data_type: 
    * For MI-CAP and MI-UAP: "shanghai" or "ucf";
    * For MI_CAP2D and MI-UAP2D: "mnist", "cifar10", "stl10"

# Citation
You can cite our paper as:
```
@article{Zhang:2023:111,
author		=	{Yu-Xuan Zhang and Hua Meng and Xue Mei Cao and Zheng Chun Zhou and Mei Yang and Avik Ranjan Adhikary},
title		=	{Interpreting vulnerabilities of multi-instance learning to adversarial perturbations},
journal		=	{{arXiv}},
pages		=	{1--11},
year		=	{2023},
url             =	{https://arxiv.org/abs/2211.17071}
```
