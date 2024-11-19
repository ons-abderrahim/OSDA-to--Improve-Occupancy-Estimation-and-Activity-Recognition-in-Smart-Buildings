from .mydataset import ImageFolder, ImageFilelist
from .unaligned_data_loader import UnalignedDataLoader
from utils.utils import *
import os
import torch
import torch.utils.data as util_data
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

class PytorchDataSet(util_data.Dataset):
    
    def __init__(self, df, len_features):
        FEATURES = list(i for i in df.columns if i!= 'labels')
        TARGET = "labels"

        from sklearn.preprocessing import StandardScaler
        Normarizescaler = StandardScaler()
        Normarizescaler.fit(np.array(df[FEATURES]))
        
        # for test data, In test data, it's easier to fill it with something on purpose.
        
        if "labels" not in df.columns:
            df["labels"] = 9999
        
        self.df = df
        self.len_features = len_features
        
        self.train_X = np.array(self.df[FEATURES])
        self.train_Y = np.array(self.df[TARGET])
        self.train_X = Normarizescaler.transform(self.train_X)
        
        
        self.train_X = torch.from_numpy(self.train_X).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()
    
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, idx):
        
        return self.train_X[idx].view(1, self.len_features), self.train_Y[idx]


def get_loader(source_path, target_path, evaluation_path, transforms, batch_size=32):
    Source_train = pd.read_csv(source_path)
    #Source_test = pd.read_csv("/content/drive/MyDrive/CLDA/data/Source_test.csv")
    Target_train = pd.read_csv(target_path)
    Target_test = pd.read_csv(evaluation_path)

    FEATURES_dset = list(i for i in Source_train.columns if i!= 'labels')
    len_features = len(FEATURES_dset)
    print(Source_train.shape)
    unique_labels = Source_train['labels'].unique().tolist()
    


    source_folder = PytorchDataSet(Source_train,len_features )
    target_folder_train = PytorchDataSet(Target_train, len_features)
    eval_folder_test = PytorchDataSet(Target_test, len_features)
    train_loader = UnalignedDataLoader()
    train_loader.initialize(source_folder, target_folder_train, batch_size)

    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1)

    return train_loader, test_loader, unique_labels, len_features


