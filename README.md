# Self-attention enhanced Spatial Temporal Graph Convolutional Network for Skeleton-based Emotion Recognition

## Introduction

S-STGCN: Self-attention enhanced Spatial Temporal Graph Convolutional Network for Skeleton-based Emotion Recognition

## Preprocessed data

Get the preprocessed skeleton data [Application Form](https://forms.gle/2snjzMrJkHPz8Pyg9).

If you want to download the preprocessed skeleton data, please ask the license to [the IEMOCAP team](https://sail.usc.edu/iemocap/index.html) first. Then contact us and attach your IEMOCAP license to the email. We will send you the password as soon as possible.

After downloading the preprocessed skeleton data, please unzip them and put them in `./data`.

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

When you use our model/code/data, please cite
```
@article{shi2021skeleton,
  title={Skeleton-Based Emotion Recognition Based on Two-Stream Self-Attention Enhanced Spatial-Temporal Graph Convolutional Network},
  author={Shi, Jiaqi and Liu, Chaoran and Ishi, Carlos Toshinori and Ishiguro, Hiroshi},
  journal={Sensors},
  volume={21},
  number={1},
  pages={205},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

## Contact

For any question, please contact ```shi.jiaqi@irl.sys.es.osaka-u.ac.jp```
