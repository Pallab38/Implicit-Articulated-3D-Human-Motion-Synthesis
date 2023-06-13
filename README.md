# Implicit-Articulated-3D-Human-Motion-Synthesis (Master's Thesis)

## Abstract 
In this thesis, we address the problem of synthesizing human motion from sparse multi-view RGB videos. This is a challenging task since reconstructing a human from a set of sparse views is quite difficult because of extreme self-occlusions and the high variance of joint locations of the human body. Moreover, the appearance information has to be extracted too. We approach these challenges by presenting a novel approach that incorporates neural radiance field (NeRF) [[1]](#1) defined in the 3D space with the SMPL body model [[2]](#2). We also utilize per-subject latent codes for incorporating human body shape and appearance information. This formulation allows us to articulate pose information for unseen poses as well. Experiments on ZJU-MoCap dataset [[3]](#3) show that our approach performs competitively with the state-of-the-art approaches in terms of novel view synthesis. Furthermore, we demonstrate the capability of our method in terms of synthesizing novel poses.
## Method
![overview](https://github.com/Pallab38/Implicit-Articulated-3D-Human-Motion-Synthesis/blob/main/resources/overview.png)

Given sampled 3D points of the human body surface along with latent codes as input, our method first regresses density and color to encode geometry and appearance of the human using two separate neural networks: density and painter network. This step represents the human body as a continuous neural implicit function. Then, the neural body scene representation is rendered by a sphere-based renderer Pulsar. Finally, a fully convolutional network is used as the neural shader module  to obtain the final RGB image and the predicted mask.

### Sampling 3D Points: 
Random 3D points are sampled from the surface of the SMPL[[2]](#2) body mesh following the approach of Osada _et al._ [[4]](#4). 

### Latent Codes: 
(i) Shape latent codes to represent the size and shape of an individual person. <br>
(ii) Latent textures embed visual attributes of the person such as color, texture, etc.

### Implicit Body Representation:
The implicit scene representation is defined by a continuous function over the 3D space. To represent the human body in an implicit way, following neurla networks are implemented:
#### 1. Density Network: 
The density network computes the density of a given 3D coordinate.
#### 2. Painter Network: 
The painter network models the color at any location as a function of that 3D coordinate, normalized view direction, and latent codes.
#### Positional Encoding: 
Gaussian Fourier feature mapping [[5]](#5) to encode 3D points and normalized view-direction to a higher-dimensional space.
### Differentiable Rendering: 


## Trained Model 
1. [Full Model (MEGA Link)](https://mega.nz/fm/ouIghJ4I)
2. [Latent Model(MEGA Link) ](https://mega.nz/fm/Q3JwDZRY)

## DATASET 
[ZJU-MoCap Dataset](https://github.com/zju3dv/EasyMocap#zju-mocap). Three subjects are chosen: (i) 313 (Arms Swing), (ii) 377 (Twirling), and (iii) 386 (Punching).

## Results: Novel View Synthesis
### Subject: 313 (Arms Swing)
![Subject: 313](https://github.com/Pallab38/Implicit-Articulated-3D-Human-Motion-Synthesis/blob/main/resources/nvs/313_nvs_20fps.gif)
### Subject: 377 (Twirling)
![Subject: 377](https://github.com/Pallab38/Implicit-Articulated-3D-Human-Motion-Synthesis/blob/main/resources/nvs/377_nvs_20fps.gif)

### Subject: 386 (Punching)
![Subject: 386](https://github.com/Pallab38/Implicit-Articulated-3D-Human-Motion-Synthesis/blob/main/resources/nvs/386_nvs_20fps.gif)

## Results: Novel Pose Synthesis
### Subject: 313 (Arms Swing)
![Subject: 313](https://github.com/Pallab38/Implicit-Articulated-3D-Human-Motion-Synthesis/blob/main/resources/nps/313_20fps.gif)
### Subject: 377 (Twirling)
![Subject: 377](https://github.com/Pallab38/Implicit-Articulated-3D-Human-Motion-Synthesis/blob/main/resources/nps/377_20fps.gif)

### Subject: 386 (Punching)
![Subject: 386](https://github.com/Pallab38/Implicit-Articulated-3D-Human-Motion-Synthesis/blob/main/resources/nps/386_20fps.gif)


## REFERENCES
<a id="1">[1]</a>
Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren (2021).
Nerf: Representing scenes as neural radiance fields for view synthesis.
Communications of the ACM.<br>

<a id="2">[2]</a>
Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J (2015).
SMPL: A skinned multi-person linear model.
ACM transactions on graphics (TOG).<br>

<a id ="3"> [3]</a>
Peng, Sida and Zhang, Yuanqing and Xu, Yinghao and Wang, Qianqian and Shuai, Qing and Bao, Hujun and Zhou, Xiaowei (2021). 
Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. <br>
<a id="4"> [4] </a> Osada, Robert and Funkhouser, Thomas and Chazelle, Bernard and Dobkin, David (2002). <br>
Shape distributions. <br>
ACM Transactions on Graphics (TOG). <br>
<a id="5"> [5] </a>  
Tancik, Matthew and Srinivasan, Pratul and Mildenhall, Ben and Fridovich-Keil, Sara and Raghavan, Nithin and Singhal, Utkarsh and Ramamoorthi, Ravi and Barron, Jonathan and Ng, Ren <br>
Fourier features let networks learn high frequency functions in low dimensional domains. <br>
Advances in Neural Information Processing Systems.

<a id="6"> [6]</a>
Lassner, Christoph and Zollhofer, Michael <br>
Pulsar: Efficient sphere-based neural rendering <br>
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
