# Self-attention enhanced Spatial Temporal Graph Convolutional Network for Skeleton-based Emotion Recognition

## Introduction

S-STGCN: Self-attention enhanced Spatial Temporal Graph Convolutional Network for Skeleton-based Emotion Recognition

## Preprocessed data

Download the preprocessed skeleton data from [Google Drive](https://drive.google.com/file/d/1yK1_o5Jv5syCafYsiwxM9XH82M1JYxLZ/view?usp=sharing). Then unzip them and put them in `./data`.

If you want to download the original IEMOCAP dataset, please submit your request to [the IEMOCAP team](https://sail.usc.edu/iemocap/index.html).

## Training

To train on the joint stream, please run

```
python train_motion.py
```

To train on the bone stream, please run

```
python train_motion.py --stream bone
```

## Contact

For any question, please contact ```shi.jiaqi@irl.sys.es.osaka-u.ac.jp```
