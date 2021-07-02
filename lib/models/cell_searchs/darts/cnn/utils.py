import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

import boto3
from os import path as osp

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
  
def save_checkpoint(state, filename):
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

def exact_hessian(network, val_loader, criterion, xloader, epoch, logger, args):
  labels = []
  try:
    for i in range(network._max_nodes):
      for n in network._op_names:
        labels.append(n + "_" + str(i))
  except Exception as e:
    print(f"Couldnt compute labels for Hessian due to {e}")

  network.logits_only=True
  val_x, val_y = next(iter(val_loader))
  val_loss = criterion(network(val_x.to('cuda')), val_y.to('cuda'))
  try:
    train_x, train_y, _, _ = next(iter(xloader))
  except:
    train_x, train_y = next(iter(xloader))

  train_loss = criterion(network(train_x.to('cuda')), train_y.to('cuda'))
  val_hessian_mat = _hessian(val_loss, network.arch_params())
  if epoch == 0:
    print(f"Example architecture Hessian: {val_hessian_mat}")
  val_eigenvals, val_eigenvecs = torch.eig(val_hessian_mat)
  try:
    if not args.merge_train_val_supernet:
      train_hessian_mat = _hessian(train_loss, network.arch_params())
      train_eigenvals, train_eigenvecs = torch.eig(train_hessian_mat)
    else:
      train_eigenvals = val_eigenvals
  except:
    train_eigenvals = val_eigenvals
  val_eigenvals = val_eigenvals[:, 0] # Drop the imaginary components
  if epoch == 0:
    print(f"Example architecture eigenvals: {val_eigenvals}")
  train_eigenvals = train_eigenvals[:, 0]
  val_dom_eigenvalue = torch.max(val_eigenvals)
  train_dom_eigenvalue = torch.max(train_eigenvals)
  eigenvalues = {"max":{}, "spectrum": {}}
  eigenvalues["max"]["train"] = train_dom_eigenvalue
  eigenvalues["max"]["val"] = val_dom_eigenvalue
  eigenvalues["spectrum"]["train"] = {k:v for k,v in zip(labels, train_eigenvals)}
  eigenvalues["spectrum"]["val"] = {k:v for k,v in zip(labels, val_eigenvals)}
  network.logits_only=False
  return eigenvalues
    
def approx_hessian(network, val_loader, criterion, xloader, args):
  network.logits_only=True
  val_eigenvals, val_eigenvecs = compute_hessian_eigenthings(network, val_loader, criterion, 1, mode="power_iter", 
                                                             power_iter_steps=50, max_samples=128, arch_only=True, full_dataset=False)
  val_dom_eigenvalue = val_eigenvals[0]
  try:
    if hasattr(args, merge_train_val_supernet) and not args.merge_train_val_supernet:
      train_eigenvals, train_eigenvecs = compute_hessian_eigenthings(network, val_loader, criterion, 1, mode="power_iter", 
                                                                    power_iter_steps=50, max_samples=128, arch_only=True, full_dataset=False)
      train_dom_eigenvalue = train_eigenvals[0]
    else:
      train_eigenvals, train_eigenvecs = None, None
      train_dom_eigenvalue = None
  except:
    train_eigenvals, train_eigenvecs, train_dom_eigenvalue = None, None, None
  eigenvalues = {"max":{}, "spectrum": {}}
  eigenvalues["max"]["val"] = val_dom_eigenvalue
  eigenvalues["max"]["train"] = train_dom_eigenvalue
  network.logits_only=False
  return eigenvalues