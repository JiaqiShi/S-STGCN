import os
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
    def __init__(self,
                 path='/home/shi/git/s-stgcn/data',
                 stream='J',
                 sub_set=0):

        self.stream = stream

        pkls = ['motion_test.pkl', 'motion_val.pkl', 'motion_test.pkl']
        pkl_path = os.path.join(path, pkls[sub_set])

        with open(pkl_path, 'rb') as f:
            self.ids, self.jointData, self.labels = pickle.load(f)

        if stream is not 'J':
            self.boneData = {}
            target_index = [i for i in range(10)]
            source_index = [0, 0, 1, 2, 2, 4, 5, 2, 7, 8]
            for k, v in self.jointData:
                self.boneData[k] = v[:, target_index, :] - v[:, source_index, :]

        self.len = len(self.ids)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        key = self.ids[index]
        if self.stream == 'J':
            return torch.FloatTensor(self.jointData[key]), torch.tensor(
                self.labels[key], dtype=torch.long), i
        elif self.stream == 'B':
            return torch.FloatTensor(self.boneData[key]), torch.tensor(
                self.labels[key], dtype=torch.long), i
        else:
            return torch.FloatTensor(self.jointData[key]), torch.FloatTensor(self.boneData[key]),\
                 torch.tensor(self.labels[key], dtype=torch.long), i
