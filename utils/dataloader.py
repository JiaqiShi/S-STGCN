import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split


def get_IEMO_dataloaders(dataset=None, batch_size=2, stream='J'):

    if dataset is None:
        dataset = IEMOCAPDataset

    train_set = dataset(sub_set=0, stream=stream)
    valid_set = dataset(sub_set=1, stream=stream)
    test_set = dataset(sub_set=2, stream=stream)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, valid_loader, test_loader

class IEMOCAPDataset(Dataset):
    def __init__(self, path='/home/shi/git/s-stgcn/data', stream='J', sub_set=0):

        pkls = ['motion_test.pkl','motion_val.pkl','motion_test.pkl']
