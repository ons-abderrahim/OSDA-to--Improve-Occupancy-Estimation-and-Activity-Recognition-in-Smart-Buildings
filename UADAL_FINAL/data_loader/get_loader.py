from .mydataset import ImageFolder, ImageFilelist
from .unaligned_data_loader import UnalignedDataLoader
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import sys
import numpy as np
from collections import Counter
#torch.set_num_threads()
import warnings
warnings.filterwarnings("ignore")

## Adopt from https://github.com/ksaito-ut/OPDA_BP/blob/master/data_loader/get_loader.py

def get_loader(source_path, target_path, evaluation_path, transforms, batch_size=32):
    sampler = None
    pin = True
    num_workers = 2

    source_folder_train = ImageFolder(os.path.join(source_path), transform=transforms[source_path])
    target_folder_train = ImageFolder(os.path.join(target_path), transform=transforms[source_path])
    target_folder_test = ImageFolder(os.path.join(evaluation_path), transform=transforms[evaluation_path])
    source_folder_test = ImageFolder(os.path.join(source_path), transform=transforms[source_path])

    freq = Counter(source_folder_train.labels)
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in source_folder_train.labels]
    sampler = WeightedRandomSampler(source_weights,
                                    len(source_folder_train.labels))
    aligned_train_loader = UnalignedDataLoader()
    aligned_train_loader.initialize(source_folder_train, target_folder_train, batch_size, sampler=sampler)

    target_train_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=pin)
    target_test_loader = torch.utils.data.DataLoader(
        target_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=pin)
    source_test_loader = torch.utils.data.DataLoader(
        source_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=pin)

    if sampler is not None:
        source_train_loader = torch.utils.data.DataLoader(source_folder_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin, drop_last=True)
    else:
        source_train_loader = torch.utils.data.DataLoader(source_folder_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True)

    return aligned_train_loader, target_train_loader, target_test_loader, source_train_loader, source_test_loader

def get_dataset_information(dataset, s_d, t_d):
    source_path = "/content/drive/MyDrive/OSDA_Papers_adaptation/UADAL_FINAL/data/Source_train.csv"
    target_path = "/content/drive/MyDrive/OSDA_Papers_adaptation/UADAL_FINAL/data/Target_train.csv"
    evaluation_data = "/content/drive/MyDrive/OSDA_Papers_adaptation/UADAL_FINAL/data/Target_test.csv"
    class_list = ['0', '1'] # To change
    num_class = len(class_list) #11############# To change in every task.

    return source_path, target_path, evaluation_data, num_class