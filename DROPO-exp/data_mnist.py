from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

class DATA_mnist(data.Dataset):
    '''
        Read data

    '''
    def __init__(self, path):
        path = 'data/MNIST/processed/training.pt'
        features_, labels = torch.load(path)
        # features_ = features_[:5000]
        # labels = labels[:5000]

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])


        features = torch.zeros((len(features_),28,28))
        for i in range(len(features_)):
            features[i] = trans(features_[i])

        features = features.numpy().reshape(-1,28*28)
        labels = labels.numpy()
     
        self.x = features
        self.y = labels

    def __getitem__(self, index):
     
        features, target = self.x[index,:], self.y[index]
        return torch.tensor(features), torch.tensor(target) 

    def __len__(self):
        return len(self.x)