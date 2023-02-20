from __future__ import print_function
import numpy as np
import torch
from torchvision import datasets
import torch.utils.data as data

import csv

class MERGE_DATA_SET(data.Dataset):

    def __init__(self, source, target, source_labels, target_labels):
        '''
        source: source dataset with original source_labels
        target: target dataset with original target_labels
        ''' 
        # index = np.random.permutation(len(source_labels)+ len(target_labels))
        self.features = data.ConcatDataset([source, target])
        self.labels = np.concatenate([source_labels, target_labels])



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.features[index][0], self.labels[index]


        return img, target

    def __len__(self):
        return len(self.features)

class WEIGHT_DATA_SET(data.Dataset):
    '''
    dataset class with instance weight 
    '''
    def __init__(self, original_data, weights,args):
        '''
        weights are same dimensional witi original data
        '''
        if args.clip_weight:
            weights_upper_bound = args.weights_upper_bound
            weights_lower_bound = args.weights_lower_bound

            weights[weights < weights_lower_bound] = weights_lower_bound
            weights[weights > weights_upper_bound] = weights_upper_bound

        # print(max(weights),min(weights))
        self.data = original_data
        self.weights = weights

    def __getitem__(self, index):

        img, target = self.data[index]
        weight = self.weights[index]

        return img, target, weight

    def __len__(self):
        return len(self.data)

class WEIGHT_DATA_SET_OBD(data.Dataset):
    '''
    dataset class with instance weight
    '''
    def __init__(self, original_data, weights,args):
        '''
        weights are same dimensional witi original data
        '''
        if args.clip_weight:
            weights_upper_bound = args.weights_upper_bound
            weights_lower_bound = args.weights_lower_bound

            weights[weights < weights_lower_bound] = weights_lower_bound
            weights[weights > weights_upper_bound] = weights_upper_bound

        # print(max(weights),min(weights))
        self.data = original_data
        self.weights = weights

    def __getitem__(self, index):

        context, action, reward = self.data[index]
        weight = self.weights[index]

        return context, action,reward, weight

    def __len__(self):
        return len(self.data)

class WEIGHT_DATA_SET_OBD(data.Dataset):
    '''
    dataset class with instance weight
    '''
    def __init__(self, original_data, weights,args):
        '''
        weights are same dimensional witi original data
        '''
        if args.clip_weight:
            weights_upper_bound = args.weights_upper_bound
            weights_lower_bound = args.weights_lower_bound

            weights[weights < weights_lower_bound] = weights_lower_bound
            weights[weights > weights_upper_bound] = weights_upper_bound

        # print(max(weights),min(weights))
        self.data = original_data
        self.weights = weights

    def __getitem__(self, index):

        context, action, reward = self.data[index]
        weight = self.weights[index]

        return context, action,reward, weight

    def __len__(self):
        return len(self.data)


class IW_DATA_SET(data.Dataset):
    '''
    dataset class with instance weight 
    '''
    def __init__(self, original_data, weights):
        '''
        weights are same dimensional witi original data
        '''
        self.data = original_data
        self.weights = weights

    def __getitem__(self, index):

        img, target = self.data[index][0] * self.weights[index], self.data[index][1]

        return img, target

    def __len__(self):
        return len(self.data)


class RE_DATA_SET(data.Dataset):
    '''
    Test dataset for regression
    '''
    def __init__(self):

        x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

        y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                            [3.366], [2.596], [2.53], [1.221], [2.827],
                            [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


        x_train = torch.from_numpy(x_train)

        y_train = torch.from_numpy(y_train)
        
        self.x = x_train
        self.y = y_train
        self.n = len(y_train)

    def __getitem__(self, index):

        img, target = self.x[index], self.y[index]

        return img, target

    def __len__(self):
        return len(self.y)

class FA_DATA_SET(data.Dataset):

    def __init__(self):
        '''
        read csv 
        '''
        reader = csv.reader(open("data/Fa_data.csv", "rb"), delimiter=",")
        x = list(reader)
        print(x[2])
        result = np.array(x[1:]).astype("float")
        m = len(result)
        print(m)
        print(np.shape(result))
        result = result[:, 1:]
        self.y = result[:, 2]
        self.x = result[:, 3:]



    def __getitem__(self, index):

        img, target = self.x[index, :], self.y[index]

        return img, target

    def __len__(self):
        return len(self.y)


