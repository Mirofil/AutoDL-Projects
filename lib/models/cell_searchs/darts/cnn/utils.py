import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

import boto3
from os import path as osp
from collections import defaultdict

def upload_to_s3(source, bucket, key):
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(source, bucket, key)

def download_from_s3(key, bucket, target):
    # Make sure directory exists before downloading to it.
    target_dir = os.path.dirname(target)
    if len(target_dir) and not os.path.exists(target_dir):
        os.makedirs(target_dir)

    s3 = boto3.resource('s3')
    try:
      s3.meta.client.download_file(bucket, key, target)
    except Exception as e:
      print(e)

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.reshape(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

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


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))
  
def save_checkpoint2(state, filename):
  if osp.isfile(filename):
    os.remove(filename)
  try:
    torch.save(state, filename.parent / (filename.name + 'tmp'))
    os.replace(filename.parent / (filename.name + 'tmp'), filename)
  except Exception as e:
    print(f"Failed to save new checkpoint into {filename} due to {e}")
  assert osp.isfile(filename), 'save filename : {:} failed, which is not found.'.format(filename)
  return filename

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def genotype_to_adjacency_list(genotype, steps=4):
  # Should pass in genotype.normal or genotype.reduce
  G = defaultdict(list)
  for nth_node, connections in enumerate(chunks(genotype, 2), start=2): # Darts always keeps two connections per node and first two nodes are fixed input
    for connection in connections:
      G[connection[1]].append(nth_node)
  # Add connections from all intermediate nodes to Output node
  for intermediate_node in [2,3,4,5]:
    G[intermediate_node].append(6)
  return G
    
def DFS(G,v,seen=None,path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths

def genotype_depth(genotype):
  # The shortest path can start in either of the two input nodes
  cand0 = max(len(p) for p in DFS(genotype_to_adjacency_list(genotype), 0))
  cand1 = max(len(p) for p in DFS(genotype_to_adjacency_list(genotype), 0))
