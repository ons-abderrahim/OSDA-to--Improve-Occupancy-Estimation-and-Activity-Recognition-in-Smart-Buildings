# -*- coding: utf-8 -*-
import os
import sys
import copy
import random
from PIL import Image
from collections import Counter
import pickle
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
# from .datasets import get_dataset, name2benchmark
from .mydataset import ImageFolder, ImageFilelist, ImageList
import utils

from utils.utils import default_loader
import torch.utils.data as util_data
import pandas as pd
import numpy as np

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

sys.path.append('../')




class PytorchDataSet(util_data.Dataset):
    
    def __init__(self, df, len_features):
        self.len_features = len_features
        FEATURES = list(i for i in df.columns if i!= 'labels')
        TARGET = "labels"

        from sklearn.preprocessing import StandardScaler
        Normarizescaler = StandardScaler()
        Normarizescaler.fit(np.array(df[FEATURES]))
        
        # for test data, In test data, it's easier to fill it with something on purpose.
        
        if "labels" not in df.columns:
            df["labels"] = 9999
        
        self.df = df
        
        self.data = np.array(self.df[FEATURES])
        self.train_Y = np.array(self.df[TARGET])
        self.data = Normarizescaler.transform(self.data)
        
        
        self.data = torch.from_numpy(self.data).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        return self.data[idx].view(1, self.len_features), self.train_Y[idx]



###### Adopt from https://github.com/virajprabhu/SENTRY/blob/main/datasets/base.py

class DatasetWithIndicesWrapper(torch.utils.data.Dataset):
    def __init__(self, data, targets, transforms, base_transforms):
        self.data = data
        self.targets = targets
        self.transforms = transforms
        self.base_transforms = base_transforms
        self.rand_aug_transforms = copy.deepcopy(self.base_transforms)
        self.committee_size = 1
        #self.ra_obj = RandAugment(1, 2.0)
        #self.rand_aug_transforms.transforms.insert(0, self.ra_obj)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        data, target = self.data[index], self.targets[index]
        #data = default_loader(self.data[index])
        #rand_aug_lst = [self.rand_aug_transforms(data) for _ in range(self.committee_size)]
        return (self.data, self.data, self.data), int(target), int(index)


class UDADataset:
    """
    Dataset Class
    """

    def __init__(self, train_path, test_path, num_classes, train_transforms, test_transforms, is_target=False, batch_size=128):
        #self.name = name
        self.is_target = is_target
        self.batch_size = batch_size
        self.train_size = None
        self.train_dataset = None
        self.num_classes = None
        self.train_transforms = None
        self.test_transforms = None
        self.train_path = train_path
        self.test_path = test_path
        self.num_classes = num_classes
        self.train_transforms=train_transforms
        self.test_transforms=test_transforms

    def get_num_classes(self):
        return self.num_classes

    def long_tail_train(self, key):
        """Manually long-tails target training set by loading checkpoint
        Args:
            key: Identifier of long-tailed checkpoint
        """
        ixs = pickle.load(open(os.path.join('checkpoints', '{}.pkl'.format(key)), 'rb'))
        self.train_dataset.data = self.train_dataset.data[ixs]
        self.train_dataset.targets = torch.from_numpy(np.array(self.train_dataset.targets)[ixs])

    def get_dsets(self):
        """Generates and return train, val, and test datasets

        Returns:
            Train, val, and test datasets.
            
        """
        Source_train = pd.read_csv(self.train_path)
        FEATURES_dset = list(i for i in Source_train.columns if i!= 'labels')
        len_features = len(FEATURES_dset)
        unique_labels = Source_train['labels'].unique().tolist()


        train_dataset = PytorchDataSet(Source_train, len_features)
        val_dataset = PytorchDataSet(Source_train, len_features)######### Need validation data.
        test_dataset = PytorchDataSet(Source_train, len_features)



        train_dataset.targets, val_dataset.targets, test_dataset.targets = train_dataset.train_Y, val_dataset.train_Y, test_dataset.train_Y


        self.train_dataset = DatasetWithIndicesWrapper(train_dataset.data, train_dataset.targets,
                                                       self.train_transforms, self.test_transforms)
        self.val_dataset = DatasetWithIndicesWrapper(val_dataset.data, val_dataset.targets,
                                                     self.test_transforms, self.test_transforms)
        self.test_dataset = DatasetWithIndicesWrapper(test_dataset.data, test_dataset.targets,
                                                      self.test_transforms, self.test_transforms)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_loaders(self, shuffle=True, num_workers=1, class_balance_train=False):
        """Constructs and returns dataloaders

        Args:
            shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.
            num_workers (int, optional): Number of threads. Defaults to 4.
            class_balance_train (bool, optional): Whether to class-balance train data loader. Defaults to False.

        Returns:
            Train, val, test dataloaders, as well as selected indices used for training
        """
        if not self.train_dataset: self.get_dsets()
        num_train = len(self.train_dataset)
        self.train_size = num_train

        train_idx = np.arange(len(self.train_dataset))
       

        train_loader = torch.utils.data.DataLoader(self.train_dataset, \
                                                   batch_size=self.batch_size, num_workers=num_workers, drop_last=True)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, shuffle=False, \
                                                 batch_size=self.batch_size)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader, train_idx
