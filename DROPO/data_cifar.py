from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torchvision import datasets
import torchvision

torch.set_default_tensor_type('torch.DoubleTensor')


class DATA_CIFAR10(data.Dataset):
    '''
        Read data

    '''

    def __init__(self, transform):
        trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data/cifar', train=False, download=True, transform=transform)
        feature_train_set = trainset.data
        feature_test_set = testset.data

        features = np.concatenate((feature_train_set, feature_test_set), axis=0)
        labels = np.concatenate((trainset.targets, testset.targets), axis=0)

        new_features = torch.zeros((len(features), 3, 32, 32))
        for i in range(len(features)):
            new_features[i] = transform(features[i])
        # self.x = new_features.reshape(-1, 32 * 32 * 3)[:100]
        # self.y = labels[:100]
        self.x = new_features.reshape(-1, 32 * 32 * 3)[:30000]
        self.y = labels[:30000]

    def __getitem__(self, index):
        features, target = self.x[index, :], self.y[index]
        return torch.tensor(features), torch.tensor(target)

    def __len__(self):
        return len(self.x)

#