# Self-attention enhanced Spatial Temporal Graph Convolutional Network for Skeleton-based Emotion Recognition

## Introduction

S-STGCN: Self-attention enhanced Spatial Temporal Graph Convolutional Network for Skeleton-based Emotion Recognition

<!-- Right now, we’re in your first GitHub **repository**. A repository is like a folder or storage space for your project. Your project's repository contains all its files such as code, documentation, images, and more. It also tracks every change that you—or your collaborators—make to each file, so you can always go back to previous versions of your project if you make any mistakes.

This repository contains three important files: The HTML code for your first website on GitHub, the CSS stylesheet that decorates your website with colors and fonts, and the **README** file. It also contains an image folder, with one image file. -->

## Preprocessed data

Download the preprocessed skeleton data from [Google Drive](https://drive.google.com/file/d/1yK1_o5Jv5syCafYsiwxM9XH82M1JYxLZ/view?usp=sharing). Then unzip them and put them in `./data`.

<!-- ## Note

Due to the random number and batch size, the results of each traning may be inconsistent. In the future, we will try to eliminate the interference of random factors. -->

## Training

To train the model, please run

```
python train_motion.py
```
<!-- 
To train on the bone stream, please run

```
python train_motion.py --stream bone
``` -->

## Contact

For any question, please contact ```shi.jiaqi@irl.sys.es.osaka-u.ac.jp```
