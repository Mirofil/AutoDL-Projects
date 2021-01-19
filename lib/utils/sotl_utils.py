import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_search_spaces
from nats_bench   import create

def simulate_train_eval_sotl(api,
                        arch,
                        dataset,
                        iepoch=None,
                        hp='12',
                        account_time=True,
                        metric='valid-accuracy',
                        e=1,
                        is_random=True):
  """This function is used to simulate training and evaluating an arch."""
  index = api.query_index_by_arch(arch)
  all_names = ('cifar10', 'cifar100', 'ImageNet16-120')
  if dataset not in all_names:
    raise ValueError('Invalid dataset name : {:} vs {:}'.format(
        dataset, all_names))
  if dataset == 'cifar10':
      dataset = 'cifar10-valid'

  if e > 1 and 'loss' in metric:
      losses = []
      for i in range(iepoch-e+1, iepoch+1):
        info = api.get_more_info(index, dataset, iepoch=i, hp=hp, is_random=is_random)
        losses.append(info[metric])

      observed_metric, time_cost = sum(losses), info['train-all-time'] + info['valid-per-time']

  else:
    info = api.get_more_info(index, dataset, iepoch=iepoch, hp=hp, is_random=is_random)
    observed_metric, time_cost = info[metric], info['train-all-time'] + info['valid-per-time']
  if metric == 'train-loss':
    observed_metric = -observed_metric
  latency = api.get_latency(index, dataset)
  if account_time:
    api._used_time += time_cost
  return observed_metric, latency, time_cost, api._used_time