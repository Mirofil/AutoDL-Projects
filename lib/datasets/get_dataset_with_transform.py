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

from .DownsampledImageNet import ImageNet16
from .SearchDatasetWrap import SearchDataset
from config_utils import load_config
from .cifar5m import Cifar5m


Dataset2Class = {'cifar10' : 10,
                 'cifar100': 100,
                 'cifar5m': 10,
                 'imagenet-1k-s':1000,
                 'imagenet-1k' : 1000,
                 'ImageNet16'  : 1000,
                 'ImageNet16-150': 150,
                 'ImageNet16-120': 120,
                 'ImageNet16-200': 200}
class CUTOUT(object):

  def __init__(self, length):
    self.length = length

  def __repr__(self):
    return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img


imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}



class Lighting(object):
  def __init__(self, alphastd,
         eigval=imagenet_pca['eigval'],
         eigvec=imagenet_pca['eigvec']):
    self.alphastd = alphastd
    assert eigval.shape == (3,)
    assert eigvec.shape == (3, 3)
    self.eigval = eigval
    self.eigvec = eigvec

  def __call__(self, img):
    if self.alphastd == 0.:
      return img
    rnd = np.random.randn(3) * self.alphastd
    rnd = rnd.astype('float32')
    v = rnd
    old_dtype = np.asarray(img).dtype
    v = v * self.eigval
    v = v.reshape((3, 1))
    inc = np.dot(self.eigvec, v).reshape((3,))
    img = np.add(img, inc)
    if old_dtype == np.uint8:
      img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype(old_dtype), 'RGB')
    return img

  def __repr__(self):
    return self.__class__.__name__ + '()'


def get_datasets(name, root, cutout, mmap=None, total_samples=None):

  print(f"Trying to retrieve dataset {name} at path {root}")

  if name == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif name == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std  = [x / 255 for x in [68.2, 65.4, 70.4]]
  elif name == "cifar5m":
    # mean/std are estimated from the first 1M samples. Should be almost the same as CIFAR10 tho
    mean = [x / 255 for x in [121.3, 119.2, 109.8]]
    std = [x / 255 for x in [63.1, 62.1, 66.6]]
  elif name.startswith('imagenet-1k'):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  elif name.startswith('ImageNet16'):
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
  else:
    raise TypeError("Unknow dataset : {:}".format(name))

  # Data Argumentation
  if name == 'cifar10' or name == 'cifar100' or name == "cifar5m":
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
    if cutout > 0 : lists += [CUTOUT(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 32, 32) if not name == "cifar5m" else (1, 32, 32, 3) 
  elif name.startswith('ImageNet16'):
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
    if cutout > 0 : lists += [CUTOUT(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 16, 16)
  elif name == 'tiered':
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(80, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
    if cutout > 0 : lists += [CUTOUT(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.CenterCrop(80), transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 32, 32)
  elif name.startswith('imagenet-1k'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if name == 'imagenet-1k':
      xlists    = [transforms.RandomResizedCrop(224)]
      xlists.append(
        transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2))
      xlists.append( Lighting(0.1))
    elif name == 'imagenet-1k-s':
      xlists    = [transforms.RandomResizedCrop(224, scale=(0.2, 1.0))]
    else: raise ValueError('invalid name : {:}'.format(name))
    xlists.append( transforms.RandomHorizontalFlip(p=0.5) )
    xlists.append( transforms.ToTensor() )
    xlists.append( normalize )
    train_transform = transforms.Compose(xlists)
    test_transform  = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    xshape = (1, 3, 224, 224)
  else:
    raise TypeError("Unknow dataset : {:}".format(name))

  if name == 'cifar10':
    train_data = dset.CIFAR10 (root, train=True , transform=train_transform, download=True)
    test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000
  elif name == 'cifar100':
    train_data = dset.CIFAR100(root, train=True , transform=train_transform, download=True)
    test_data  = dset.CIFAR100(root, train=False, transform=test_transform , download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000
  elif name == "cifar5m":
    train_data = Cifar5m(root, train = True, transform=train_transform, mmap=mmap, total_samples=total_samples)
    
    test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)

  elif name.startswith('imagenet-1k'):
    train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
    test_data  = dset.ImageFolder(osp.join(root, 'val'),   test_transform)
    assert len(train_data) == 1281167 and len(test_data) == 50000, 'invalid number of images : {:} & {:} vs {:} & {:}'.format(len(train_data), len(test_data), 1281167, 50000)
  elif name == 'ImageNet16':
    train_data = ImageNet16(root, True , train_transform)
    test_data  = ImageNet16(root, False, test_transform)
    assert len(train_data) == 1281167 and len(test_data) == 50000
  elif name == 'ImageNet16-120':
    train_data = ImageNet16(root, True , train_transform, 120)
    test_data  = ImageNet16(root, False, test_transform , 120)
    assert len(train_data) == 151700 and len(test_data) == 6000
  elif name == 'ImageNet16-150':
    train_data = ImageNet16(root, True , train_transform, 150)
    test_data  = ImageNet16(root, False, test_transform , 150)
    assert len(train_data) == 190272 and len(test_data) == 7500
  elif name == 'ImageNet16-200':
    train_data = ImageNet16(root, True , train_transform, 200)
    test_data  = ImageNet16(root, False, test_transform , 200)
    assert len(train_data) == 254775 and len(test_data) == 10000
  else: raise TypeError("Unknow dataset : {:}".format(name))
  
  class_num = Dataset2Class[name]
  return train_data, test_data, xshape, class_num



class SubsetSequentialSampler(Sampler):
    #https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#SubsetRandomSampler
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, indices, epochs, extra_split=False, shuffle=True):
        self.indices = indices
        if not extra_split:
          if shuffle:
            self.all_indices = [torch.randperm(len(indices)) for _ in range(epochs)]
          else:
            raise NotImplementedError # Doesnt make sense to go in this branch
        else:

          if shuffle:
            permuted_indices = torch.chunk(torch.randperm(len(indices)), epochs)
          else:
            permuted_indices = torch.chunk(torch.tensor(range(len(indices))), epochs)
          self.all_indices = permuted_indices

        self.epochs = epochs
        self.counter = 0

    def __iter__(self):

        return (self.indices[i] for i in self.all_indices[self.counter % self.epochs])

    def __len__(self) -> int:
        return len(self.all_indices[0])

    def reset_counter(self):
      self.counter = 0

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == class_name:
            indices.append(i)
    return indices

def get_nas_search_loaders(train_data, valid_data, dataset, config_root, batch_size, workers, valid_ratio=1, 
  determinism =None, meta_learning=False, epochs=1, merge_train_val=False, merge_train_val_and_use_test=False, extra_split=True):
  #NOTE It is NECESSARY not to return anything using valid_data here! The valid_data is the true test set
  if valid_ratio < 1 and dataset != "cifar10":
    raise NotImplementedError
  
  if isinstance(batch_size, (list,tuple)):
    batch, test_batch = batch_size
  else:
    batch, test_batch = batch_size, batch_size
  if dataset == 'cifar10':
    #split_Fpath = 'configs/nas-benchmark/cifar-split.txt'

    if meta_learning == "all":
      train_classes = [0,2,5,7,8]
      valid_classes = [1,3,4,6,9]
      train_split = list(itertools.chain.from_iterable([get_indices(train_data, class_idx) for class_idx in train_classes]))
      valid_split = list(itertools.chain.from_iterable([get_indices(train_data, class_idx) for class_idx in valid_classes]))
    else:
      cifar_split = load_config('{:}/cifar-split.txt'.format(config_root), None, None)
      train_split, valid_split = cifar_split.train, cifar_split.valid # search over the proposed training and validation set
    
    if merge_train_val or merge_train_val_and_use_test:
      # For SOTL, we might want to merge those two to achieve the ultimate performance
      train_split = train_split + valid_split 
      valid_split = train_split
      if merge_train_val_and_use_test:
        # TODO I think this is not obvious here because the actual test data is in valid_data and train/valid_split do not use any of that either, but then the Test data usage is further down
        print(f"WARNING - Using CIFAR10 test set for evaluating the correlations! Now train_split (len={len(train_split)}) and valid_split (len={len(valid_split)})")

    if valid_ratio < 1:
      if not merge_train_val or merge_train_val_and_use_test:
        valid_split = random.sample(valid_split, math.floor(len(valid_split)*valid_ratio))
      else:
        # Note that in this branch, train_split and valid_split are both the 50k samples of training CIFAR10
        assert len(train_split) == len(valid_split)
        train_split, valid_split = train_split[:round((1-valid_ratio)*len(train_split))], train_split[round((1-valid_ratio)*len(train_split)):]
        print(f"Train_split after valid_ratio has len={len(train_split)}, valid_split has len={len(train_split)}")
        assert len(set(train_split).intersection(set(valid_split))) == 0

    xvalid_data  = deepcopy(train_data)
    if hasattr(xvalid_data, 'transforms'): # to avoid a print issue
      xvalid_data.transforms = valid_data.transform
    xvalid_data.transform  = deepcopy( valid_data.transform)
    search_data   = SearchDataset(dataset, train_data, train_split, valid_split)

    print(f"""Loaded dataset {dataset} using valid split (len={len(valid_split)}), train split (len={len(train_split)}), 
      their intersection length = {len(set(valid_split).intersection(set(train_split)))}. Original data has train_data (len={len(train_data)}), 
      valid_data (CAUTION: this is not the same validation set as used for training but the test set!) (len={len(valid_data)}), search_data (len={len(search_data)})""")
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split), num_workers=workers, pin_memory=True)
    train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, 
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split) if determinism not in ['train', 'all'] else SubsetSequentialSampler(indices=train_split, epochs=epochs), num_workers=workers, pin_memory=True)
    if not merge_train_val_and_use_test:
      valid_loader  = torch.utils.data.DataLoader(xvalid_data, batch_size=test_batch, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split) if determinism not in ['val', 'all'] else SubsetSequentialSampler(indices=valid_split, epochs=epochs), num_workers=workers, pin_memory=True)
    else:
      valid_loader  = torch.utils.data.DataLoader(valid_data, batch_size=test_batch, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(range(len(valid_data))) if determinism not in ['val', 'all'] else SubsetSequentialSampler(indices=range(len(valid_data)), epochs=epochs), num_workers=workers, pin_memory=True)
  elif dataset == 'cifar5m':
    indices = list(range(len(train_data)))
    # train_split, valid_split = sklearn.model_selection.train_test_split(indices, train_size=0.5, random_state=42)
    train_split, valid_split = list(range(0,len(indices),2)), list(range(1, len(indices), 2)) # Since the CIfar5M dataset is memory mapped, we need to maintain locality of accesses

    # xvalid_data  = deepcopy(train_data)
    # if hasattr(xvalid_data, 'transforms'): # to avoid a print issue
    #   xvalid_data.transforms = valid_data.transform
    # xvalid_data.transform  = deepcopy( valid_data.transform )
    search_data   = SearchDataset(dataset, train_data, train_split, valid_split, direct_index=True, true_length = int(round(len(train_split)/epochs)))

    print(f"Train data length {len(train_data)}")

    print(f"""Loaded dataset {dataset} using valid split (len={len(valid_split)}), train split (len={len(train_split)}), 
      their intersection = {set(valid_split).intersection(set(train_split))}. Original data has train_data (len={len(train_data)}), 
      valid_data (CAUTION: this is not the same validation set as used for training but the test set!) (len={len(valid_data)}), search_data (len={len(search_data)})""")
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split) if determinism not in ['train', 'all'] else SubsetSequentialSampler(indices=train_split, epochs=epochs, extra_split=True, shuffle=False),
       num_workers=workers, pin_memory=True)
    train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, 
      sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split) if determinism not in ['train', 'all'] else SubsetSequentialSampler(indices=train_split, epochs=epochs, extra_split=extra_split, shuffle=False if extra_split else True), num_workers=workers, pin_memory=True)
    valid_loader  = torch.utils.data.DataLoader(train_data, batch_size=test_batch, 
      sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split) if determinism not in ['val', 'all'] else SubsetSequentialSampler(indices=valid_split, epochs=epochs, extra_split=extra_split, shuffle=False if extra_split else True), num_workers=workers, pin_memory=True)
  
  elif dataset == 'cifar100':
    cifar100_test_split = load_config('{:}/cifar100-test-split.txt'.format(config_root), None, None)
    search_train_data = train_data
    search_valid_data = deepcopy(valid_data) ; search_valid_data.transform = train_data.transform

    if merge_train_val or merge_train_val_and_use_test:
      search_data   = SearchDataset(dataset, [search_train_data, search_train_data], list(range(len(search_train_data))), list(range(len(search_train_data))))
    else:
      search_data   = SearchDataset(dataset, [search_train_data,search_valid_data], list(range(len(search_train_data))), cifar100_test_split.xvalid)

    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
    train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, 
      sampler=torch.utils.data.sampler.SubsetRandomSampler(range(len(train_data))) if determinism not in ['train', 'all'] else SubsetSequentialSampler(indices=range(len(train_data)), epochs=epochs), num_workers=workers, pin_memory=True)
    
    if merge_train_val or merge_train_val_and_use_test:
      valid_loader  = torch.utils.data.DataLoader(train_data, batch_size=test_batch, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(range(len(train_data))) if determinism not in ['val', 'all'] else SubsetSequentialSampler(indices=range(len(train_data)), epochs=epochs), num_workers=workers, pin_memory=True)     
    else:
      valid_loader  = torch.utils.data.DataLoader(valid_data, batch_size=test_batch, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar100_test_split.xvalid) if determinism not in ['val', 'all'] else SubsetSequentialSampler(indices=cifar100_test_split.xvalid, epochs=epochs), num_workers=workers, pin_memory=True)  
  
  elif dataset == 'ImageNet16-120':
    imagenet_test_split = load_config('{:}/imagenet-16-120-test-split.txt'.format(config_root), None, None)
    search_train_data = train_data
    search_valid_data = deepcopy(valid_data) ; search_valid_data.transform = train_data.transform
    if merge_train_val or merge_train_val_and_use_test:
      search_data   = SearchDataset(dataset, [search_train_data, search_train_data], list(range(len(search_train_data))), list(range(len(search_train_data))))
    else:
      search_data   = SearchDataset(dataset, [search_train_data, search_valid_data], list(range(len(search_train_data))), imagenet_test_split.xvalid)
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
    train_loader  = torch.utils.data.DataLoader(train_data , batch_size=batch, 
      sampler=torch.utils.data.sampler.SubsetRandomSampler(range(len(train_data))) if determinism not in ['train', 'all'] else SubsetSequentialSampler(indices=range(len(train_data)), epochs=epochs), num_workers=workers, pin_memory=True)
    
    if merge_train_val or merge_train_val_and_use_test:
      valid_loader  = torch.utils.data.DataLoader(train_data, batch_size=test_batch, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(range(len(train_data))) if determinism not in ['val', 'all'] else SubsetSequentialSampler(indices=range(len(train_data)), epochs=epochs), num_workers=workers, pin_memory=True)    
    else:
      valid_loader  = torch.utils.data.DataLoader(valid_data, batch_size=test_batch, 
        sampler=torch.utils.data.sampler.SubsetRandomSampler(imagenet_test_split.xvalid) if determinism not in ['val', 'all'] else SubsetSequentialSampler(indices=imagenet_test_split.xvalid, epochs=epochs), num_workers=workers, pin_memory=True)  
  else:
    raise ValueError('invalid dataset : {:}'.format(dataset))

  try:
    print(f"Final dataloaders {dataset} using valid split (len={len(valid_loader)}) with {valid_loader.sampler}, train split (len={len(train_loader)}) with {train_loader.sampler}, search_loader (len={len(search_loader)}) with {search_loader.sampler}")
  except:
    print("Final dataloaders do not implement _len_")
  return search_loader, train_loader, valid_loader

#if __name__ == '__main__':
#  train_data, test_data, xshape, class_num = dataset = get_datasets('cifar10', '/data02/dongxuanyi/.torch/cifar.python/', -1)
#  import pdb; pdb.set_trace()
