##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, torch
import os.path as osp
import numpy as np
import random
import math
import torchvision.datasets as dset
import torchvision.transforms as transforms
import itertools
from torch.utils.data.sampler import Sampler
from copy import deepcopy
from PIL import Image
import multiprocessing
import torch.utils.data as data

class Cifar5m(data.Dataset):
  def __init__(self, root, train, transform, train_files=[0,1,2,3,4], val_files=[5], use_num_of_class_only=None):
    self.root      = root
    self.transform = transform
    self.train     = train  # training set or valid set

    self.train_list = [
        f'cifar5m_part{i}' for i in train_files
      ]
    self.valid_list = [
        f'cifar5m_part{i}' for i in val_files
    ]
    downloaded_list = self.train_list if train else self.valid_list

    self.data    = []
    self.targets = []
  
    # now load the picked numpy arrays
    for i, file_name in enumerate(downloaded_list):
      file_path = os.path.join(self.root, file_name)
      if not os.path.exists(file_path+"X.npy"):
          print(f"Skipping {file_path} for the train={train} dataset because it was not found")
      else:
        print(f"Loading {file_path} to construct the Cifar5M dataset for train={train}")
        x = np.load(file_path + "X.npy", mmap_mode="r")
        y = np.load(file_path + "Y.npy")
        self.data.append(x)
        self.targets.append(y)

    self.data = data.ConcatDataset(self.data)
    self.targets = data.ConcatDataset(self.targets)

    # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    # if use_num_of_class_only is not None:
    #   assert isinstance(use_num_of_class_only, int) and use_num_of_class_only > 0 and use_num_of_class_only < 1000, 'invalid use_num_of_class_only : {:}'.format(use_num_of_class_only)
    #   new_data, new_targets = [], []
    #   for I, L in zip(self.data, self.targets):
    #     if 1 <= L <= use_num_of_class_only:
    #       new_data.append( I )
    #       new_targets.append( L )
      # self.data    = new_data
      # self.targets = new_targets

  def __repr__(self):
    return ('{name}({num} images, {classes} classes)'.format(name=self.__class__.__name__, num=len(self.data), classes=len(set(self.targets))))

  def __getitem__(self, index):

    img, target = self.data[index], self.targets[index]

    img = Image.fromarray(np.array(img, dtype=np.uint8))

    if self.transform is not None:
      img = self.transform(img)

    return img, target

  def __len__(self):
    return len(self.data)

def process_cifar5m(fpath, fpath_save=None):
  # fpath should be the .npz file (this cannot be mmaped) and the fpath_save should be into a .npy file (since this one can be memmapped)

  # "/storage/AutoDL-Projects/$TORCH_HOME/cifar.python/cifar5m_part0.npz"
  # process_cifar5m("/storage/AutoDL-Projects/$TORCH_HOME/cifar.python/cifar5m_part0.npz", "/storage/AutoDL-Projects/$TORCH_HOME/cifar.python/cifar5m_part0.npy")
  if fpath_save is None:
    fpath_save = fpath[0:-1]+"y"

  files = np.load(fpath)
  X = files["X"]
  # The file starts in NHWC AND we do not want NCHW! In the ImageNet16, I saw that they are in fact converting to NHWC ...
  # X = np.swapaxes(X, 3, 2)
  # X = np.swapaxes(X, 2, 1)

  Y = files["Y"]
  with open(fpath_save, "wb"):
    np.save(fpath_save[:-4]+"X"+".npy", X)
    np.save(fpath_save[:-4]+"Y"+".npy", Y)
  print(f"Processed file {fpath} into .npy format")

