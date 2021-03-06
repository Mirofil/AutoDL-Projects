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
import time

class Cifar5m(data.Dataset):
  def __init__(self, root, train, transform, train_files=[0,1,2,3,4,5], val_files=[5], use_num_of_class_only=None, total_samples=None, mmap=None):
    self.root      = root
    self.transform = transform
    self.train     = train  # training set or valid set

    self.cur_file_index = 0
    self.mmap = mmap

    self.train_list = [
        f'cifar5m_part{i}' for i in train_files
      ]
    self.valid_list = [
        f'cifar5m_part{i}' for i in val_files
    ]
    self.downloaded_list = self.train_list if train else self.valid_list

    self.data_raw    = []
    self.targets_raw = []
  
    # now load the picked numpy arrays
    done = False
    for i, file_name in enumerate(self.downloaded_list):
      if done:
        break
      file_path = os.path.join(self.root, file_name)
      if not os.path.exists(file_path+"X.npy"):
          print(f"Skipping {file_path} for the train={train} dataset because it was not found")
      else:
        print(f"Loading {file_path} to construct the Cifar5M dataset for train={train}")
        if i == 0 or mmap == "load_all" or mmap == "r":
          x = np.load(file_path + "X.npy", mmap_mode=mmap)
          y = np.load(file_path + "Y.npy")
          self.data_raw.append(x)
          self.targets_raw.append(y)
        else:
          self.data_raw.append([0 for _ in range(len(self.data_raw[0]))])
          self.targets_raw.append([0 for _ in range(len(self.data_raw[0]))])
      
      if total_samples is not None:
        if sum([len(x) for x in self.data_raw]) > total_samples:
          cur_len = 0
          for i in range(len(self.data_raw)):
            if cur_len + len(self.data_raw[i]) > total_samples:
              self.data_raw[i] = self.data_raw[i][:total_samples-cur_len]
            print(f"Picked {total_samples} samples by using up to file {file_path}")
            done = True
            break

    self.dataset = data.ConcatDataset(self.data_raw)
    self.targets = data.ConcatDataset(self.targets_raw)

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
  
  def load_next(self, i):
    print(f"Changing Cifar5m file! Previously used index {i-1}, now using {i}")
    i = i % len(self.downloaded_list)
    self.data_raw[(i-1) % len(self.downloaded_list)] = [0 for _ in range(len(self.data_raw[0]))]

    file_name = self.downloaded_list[i]
    file_path = os.path.join(self.root, file_name)
    x = np.load(file_path + "X.npy", mmap_mode=self.mmap)
    y = np.load(file_path + "Y.npy")
    self.data_raw[i] = x
    self.targets_raw[i] = y
    self.cur_file_index = i % len(self.downloaded_list)

    self.dataset = data.ConcatDataset(self.data_raw)
    self.targets = data.ConcatDataset(self.targets_raw)

  def __repr__(self):
    return ('{name}({num} images, {classes} classes)'.format(name=self.__class__.__name__, num=len(self.data_raw), classes=len(set(self.targets_raw[0]))))

  def __getitem__(self, index):
    if type(self.dataset[index]) is int and self.mmap is None:
      print(f"Trying to access {index} which is {self.dataset[index]} and beyond the currently loaded data shard {self.cur_file_index} with len={len(self.data_raw[self.cur_file_index])}; however, the dataset has length {len(self.dataset)} in total")
      self.load_next(self.cur_file_index+1)
    # print(f"Started getting item at {time.time()}")
    try:
      img, target = self.dataset[index], self.targets[index]
    # print(f"Finished getting item at {time.time()}")
    except Exception as e:
      print(index)
      print(e)

    try:
      img = Image.fromarray(np.array(img, dtype=np.uint8))
    except Exception as e:
      print(img)
      print(index)
      print(e)

    if self.transform is not None:
      img = self.transform(img)

    return img, target

  def __len__(self):
    return len(self.dataset)

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

