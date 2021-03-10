import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets import get_datasets, SearchDataset
from procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from utils import get_model_infos, obtain_accuracy
from log_utils import AverageMeter, time_string, convert_secs2time
from models import get_search_spaces
from nats_bench import create
from typing import *
import wandb
import itertools
import scipy.stats


def checkpoint_arch_perfs(archs, arch_metrics, epochs, steps_per_epoch, checkpoint_freq = None):
  """ Outputs dict of shape {counter -> List of values (order unimportant)}
  """
  checkpoints = {}
  counter = 0
  if checkpoint_freq is None:
    checkpoint_freq = max(round(steps_per_epoch / 5), 1)
  for epoch_idx in range(epochs):
    for batch_idx in range(steps_per_epoch):
      if not counter % checkpoint_freq == 0:
        counter += 1
        continue

      if counter not in checkpoints.keys():
        checkpoints[counter] = []
      for arch in archs:
        arch = arch.tostr() if type(arch) is not str else arch
        checkpoints[counter].append(arch_metrics[arch][epoch_idx][batch_idx]) # we do counter-1 because we increment it early in the loop
      
      counter += 1

  return checkpoints

def get_true_rankings(archs, api, hp='200'):
  final_accs = {genotype:summarize_results_by_dataset(genotype, api, separate_mean_std=False, hp=hp) for genotype in archs}
  true_rankings = {}
  for dataset in final_accs[archs[0]].keys():
    acc_on_dataset = [{"arch":arch.tostr(), "metric": final_accs[arch][dataset]} for i, arch in enumerate(archs)]
    acc_on_dataset = sorted(acc_on_dataset, key=lambda x: x["metric"], reverse=True)

    true_rankings[dataset] = acc_on_dataset
  
  return true_rankings, final_accs

def calc_corrs_val(archs, valid_accs, final_accs, true_rankings, corr_funs):
  #TODO this thing is kind of legacy and quite monstrous
  corr_per_dataset = {}
  for dataset in final_accs[archs[0]].keys():
    ranking_pairs = []
    for val_acc_ranking_idx, archs_idx in enumerate(np.argsort(-1*np.array(valid_accs))):
      arch = archs[archs_idx].tostr()
      for true_ranking_dict in [tuple2 for tuple2 in true_rankings[dataset]]:
        if arch == true_ranking_dict["arch"]:
          ranking_pairs.append((valid_accs[val_acc_ranking_idx], true_ranking_dict["metric"]))
          break

    ranking_pairs = np.array(ranking_pairs)
    corr_per_dataset[dataset] = {method:fun(ranking_pairs[:, 0], ranking_pairs[:, 1]) for method, fun in corr_funs.items()}
    
  return corr_per_dataset

def avg_nested_dict(d):
  # https://stackoverflow.com/questions/57311453/calculate-average-values-in-a-nested-dict-of-dicts
  try:
    d = list(d.values()) # executed during the first recursive call only
  except: 
    pass # we get into this branch on recursive calls
  _data = sorted([i for b in d for i in b.items()], key=lambda x:x[0])
  _d = [(a, [j for _, j in b]) for a, b in itertools.groupby(_data, key=lambda x:x[0])]
  return {a:avg_nested_dict(b) if isinstance(b[0], dict) else round(sum(b)/float(len(b)), 1) for a, b in _d}

def calc_corrs_after_dfs(epochs:int, xloader, steps_per_epoch:int, metrics_depth_dim, final_accs, archs, true_rankings, prefix, api, corr_funs=None, wandb_log=False, corrs_freq=4):
  # NOTE this function is useful for the sideffects of logging to WANDB
  # xloader should be the same dataLoader used to train since it is used here only for to reproduce indexes used in training. TODO we dont need both xloader and steps_per_epoch necessarily
  if corrs_freq is None:
    corrs_freq = 1
  if corr_funs is None:
    corr_funs = {"kendall": lambda x,y: scipy.stats.kendalltau(x,y).correlation, 
      "spearman":lambda x,y: scipy.stats.spearmanr(x,y).correlation, 
      "pearson":lambda x, y: scipy.stats.pearsonr(x,y)[0]}

  sotl_rankings = []
  for epoch_idx in range(epochs):
    rankings_per_epoch = []
    for batch_idx, data in enumerate(xloader):
      if (steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx > steps_per_epoch:
        break
      relevant_sotls = [{"arch": arch, "metric": metrics_depth_dim[arch][epoch_idx][batch_idx]} for i, arch in enumerate(metrics_depth_dim.keys())]
      #NOTE we need this sorting because we query the top1/top5 perf later down the line...
      vals = np.array([x["metric"] for x in relevant_sotls])
      idxs = np.argpartition(vals, kth=min(5, len(vals)-1))
      # relevant_sotls = sorted(relevant_sotls, key=lambda x: x["metric"], reverse=True) # This sorting takes 50% of total time - the code in the for loops takes miliseconds though it repeats a lot
      relevant_sotls = [relevant_sotls[idx] for idx in idxs]

      rankings_per_epoch.append(relevant_sotls)

    sotl_rankings.append(rankings_per_epoch)
   
  corrs = []
  to_log = [[] for _ in range(epochs)]
  true_step = 0
  for epoch_idx in range(epochs):
    corrs_per_epoch = []
    for batch_idx, data in enumerate(xloader):
      if (steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx > steps_per_epoch:
        break
      if batch_idx % corrs_freq != 0:
        continue

      corr_per_dataset = {}
      for dataset in final_accs[archs[0]].keys(): # the dict keys are all Dataset names
        ranking_pairs = [] # Ranking pairs do not necessarily have to be sorted. The scipy correlation routines sort it either way

        hash_index = {(str(true_ranking_dict["arch"]) if type(true_ranking_dict["arch"]) is str else true_ranking_dict["arch"].tostr()):true_ranking_dict['metric'] for pos, true_ranking_dict in enumerate(true_rankings[dataset])}
        for sotl_dict in [tuple2 for tuple2 in sotl_rankings[epoch_idx][batch_idx]]: #See the relevant_sotls instantiation 
          arch, sotl_metric = sotl_dict["arch"], sotl_dict["metric"]

          true_ranking_idx = hash_index[arch if type(arch) is str else arch.tostr()]
          ranking_pairs.append((sotl_metric, true_ranking_idx))

        ranking_pairs = np.array(ranking_pairs)
        corr_per_dataset[dataset] = {method:fun(ranking_pairs[:, 0], ranking_pairs[:, 1]) for method, fun in corr_funs.items()}
      top1_perf = summarize_results_by_dataset(sotl_rankings[epoch_idx][batch_idx][0]["arch"], api, separate_mean_std=False)
      top5 = {nth_top: summarize_results_by_dataset(sotl_rankings[epoch_idx][batch_idx][nth_top]["arch"], api, separate_mean_std=False) 
        for nth_top in range(min(5, len(sotl_rankings[epoch_idx][batch_idx])))}
      top5_perf = avg_nested_dict(top5)
      if wandb_log:
        wandb.log({prefix:{**corr_per_dataset, "top1":top1_perf, "top5":top5_perf, "batch": batch_idx, "epoch":epoch_idx}, "true_step_corr":true_step})
      to_log[epoch_idx].append({prefix:{**corr_per_dataset, "top1":top1_perf, "top5":top5_perf, "batch": batch_idx, "epoch":epoch_idx}, "true_step_corr":true_step})
      corrs_per_epoch.append(corr_per_dataset)
      
      true_step += corrs_freq


    corrs.append(corrs_per_epoch)
  
  return corrs, to_log

class ValidAccEvaluator:
  def __init__(self, valid_loader, valid_loader_iter=None):
    self.valid_loader = valid_loader
    self.valid_loader_iter=valid_loader_iter
    super().__init__()

  def evaluate(self, arch, network, criterion):
    network.eval()
    sampled_arch = arch
    with torch.no_grad():
      network.set_cal_mode('dynamic', sampled_arch)
      try:
        inputs, targets = next(self.valid_loader_iter)
      except:
        self.valid_loader_iter = iter(self.valid_loader)
        inputs, targets = next(self.valid_loader_iter)
      _, logits = network(inputs.cuda(non_blocking=True))
      loss = criterion(logits, targets.cuda(non_blocking=True))
      val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
      val_acc_top1 = val_top1.item()
      val_acc_top5 = val_top5.item()

    network.train()
    return val_acc_top1, val_acc_top5, loss.item()


# def calculate_valid_acc_single_arch(valid_loader, arch, network, criterion, valid_loader_iter=None):
#   if valid_loader_iter is None:
#     loader_iter = iter(valid_loader)
#   else:
#     loader_iter = valid_loader_iter
#   network.eval()
#   sampled_arch = arch
#   with torch.no_grad():
#     network.set_cal_mode('dynamic', sampled_arch)
#     try:
#       inputs, targets = next(loader_iter)
#     except:
#       loader_iter = iter(valid_loader)
#       inputs, targets = next(loader_iter)
#     _, logits = network(inputs.cuda(non_blocking=True))
#     loss = criterion(logits, targets.cuda(non_blocking=True))
#     val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
#     val_acc_top1 = val_top1.item()
#     val_acc_top5 = val_top5.item()

#   network.train()
#   return val_acc_top1, val_acc_top5, loss.item()

def calculate_valid_accs(xloader, archs, network):
  valid_accs = []
  loader_iter = iter(xloader)
  network.eval()
  with torch.no_grad():
    for i, sampled_arch in enumerate(archs):
      network.set_cal_mode('dynamic', sampled_arch)
      try:
        inputs, targets = next(loader_iter)
      except:
        loader_iter = iter(xloader)
        inputs, targets = next(loader_iter)
      _, logits = network(inputs.cuda(non_blocking=True))
      val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
      valid_accs.append(val_top1.item())
  network.train()
  return valid_accs

def wandb_auth(fname: str = "nas_key.txt"):
  gdrive_path = "/content/drive/MyDrive/colab/wandb/nas_key.txt"
  if "WANDB_API_KEY" in os.environ:
      wandb_key = os.environ["WANDB_API_KEY"]
  elif os.path.exists(os.path.abspath("~" + os.sep + ".wandb" + os.sep + fname)):
      # This branch does not seem to work as expected on Paperspace - it gives '/storage/~/.wandb/nas_key.txt'
      print("Retrieving WANDB key from file")
      f = open("~" + os.sep + ".wandb" + os.sep + fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists("/root/.wandb/"+fname):
      print("Retrieving WANDB key from file")
      f = open("/root/.wandb/"+fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key

  elif os.path.exists(
      os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname
  ):
      print("Retrieving WANDB key from file")
      f = open(
          os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname,
          "r",
      )
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists(gdrive_path):
      print("Retrieving WANDB key from file")
      f = open(gdrive_path, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  wandb.login()


def simulate_train_eval_sotl_whole_history(api, arch, dataset:str, 
  hp:str, account_time:bool=True, metric:str='valid-accuracy', e:int=1, iepoch=None, is_random:bool=True, wandb_log=True):
  max_epoch = 200 if hp == '200' else 12

  observed_metrics, time_costs = [], []
  for epoch_idx in range(max_epoch):
    observed_metric, latency, time_cost, total_time_cost = simulate_train_eval_sotl(api=api, 
      arch=arch, dataset=dataset, hp=hp, iepoch=epoch_idx, account_time=account_time, 
      e=e, metric=metric, is_random=is_random)
    observed_metrics.append(observed_metric)
    if wandb_log:
      wandb.log({metric:observed_metric, "true_step":epoch_idx})
    time_costs.append(time_cost)

  return observed_metrics, time_costs



def simulate_train_eval_sotl(
    api,
    arch,
    dataset,
    iepoch: Optional[int] = None,
    hp: str = "12",
    account_time: bool = True,
    metric: str = "valid-accuracy",
    e: int = 1,
    is_random: bool = True,
):
    """This function is used to simulate training and evaluating an arch."""
    index = api.query_index_by_arch(arch)
    all_names = ("cifar10", "cifar100", "ImageNet16-120", "cifar10-valid")
    if dataset not in all_names:
        raise ValueError("Invalid dataset name : {:} vs {:}".format(dataset, all_names))

    if dataset == "cifar10": # TODO I think this is not great in hindsight? But it seems to be there by design in NASBench code already. Does not make much sense to have an extra valid set here regardless
        dataset = "cifar10-valid"

    if e > 1:
        losses = []
        for i in range(max(iepoch - e + 1, 0), iepoch + 1): # Sum up the train losses over multiple preceding epochs
            info = api.get_more_info(
                index, dataset, iepoch=i, hp=hp, is_random=is_random
            )
            losses.append(info[metric])

        observed_metric, time_cost = (
            sum(losses),
            info["train-all-time"] + info["valid-per-time"],
        )

    else:
        info = api.get_more_info(
            index, dataset, iepoch=iepoch, hp=hp, is_random=is_random
        )
        observed_metric, time_cost = (
            info[metric],
            info["train-all-time"] + info["valid-per-time"],
        )
    if metric in ["train-loss", "valid-loss"]:
        observed_metric = -observed_metric
    latency = api.get_latency(index, dataset)
    if account_time:
        api._used_time += time_cost
    return observed_metric, latency, time_cost, api._used_time


def query_all_results_by_arch(
    arch: str,
    api,
    iepoch: bool = 11,
    hp: str = "12",
    is_random: bool = True,
    accs_only: bool = True,
):
    index = api.query_index_by_arch(arch)
    datasets = ["cifar10", "cifar10-valid", "cifar100", "ImageNet16-120"]
    results = {dataset: {} for dataset in datasets}
    for dataset in datasets:
        results[dataset] = api.get_more_info(
            index, dataset, iepoch=iepoch, hp=hp, is_random=is_random
        )
    if accs_only is True:
        for dataset in datasets:
            if (
                "test-accuracy" in results[dataset].keys()
            ):  # Actually it seems all the datasets have this field?
                results[dataset] = results[dataset]["test-accuracy"]
            else:
                results[dataset] = results[dataset]["valtest-accuracy"]
    return results


def summarize_results_by_dataset(genotype: str = None, api=None, results_summary=None, separate_mean_std=False, iepoch=None, hp = '200') -> dict:
  if hp == '200' and iepoch is None:
    iepoch = 199
  elif hp == '12' and iepoch is None:
    iepoch = 11

  if results_summary is None:
    abridged_results = query_all_results_by_arch(genotype, api, iepoch=iepoch, hp=hp)
    results_summary = [abridged_results] # ?? What was I trying to do here
  else:
    assert genotype is None
  interim = {}
  for dataset in results_summary[0].keys():

    if separate_mean_std:
        interim[dataset]= {"mean":round(sum([result[dataset] for result in results_summary])/len(results_summary), 2),
        "std": round(np.std(np.array([result[dataset] for result in results_summary])), 2)}
    else:
        interim[dataset] = round(sum([result[dataset] for result in results_summary])/len(results_summary), 2)
  return interim


class RecordedMetric:
  def __init__(self, name, e, return_fn):
    self.name = name
    self.e = e
    if return_fn == 'sum':
      self.return_fn = sum
    elif return_fn == "last":
      self.return_fn = lambda x: x[-1]

class SumOfWhatever:
  def __init__(self, measurements=None, e = 1, epoch_steps=None, mode="sum"):
    if measurements is None:
      self.measurements = []
      self.measurements_flat = []
    else:
      self.measurements = measurements
      self.measurements_flat = list(itertools.chain.from_iterable(measurements))
    self.epoch_steps = epoch_steps
    self.e =e
    self.mode = mode

  def update(self, epoch, val):

    while epoch >= len(self.measurements):
      self.measurements.append([])
    self.measurements[epoch].append(val)
    self.measurements_flat.append(val)

  def get_time_series(self, e=None, mode=None, window_size = None, chunked=False):
    if mode is None:
      mode = self.mode

    params = self.guess(e=e, mode=mode, epoch_steps=None)
    return_fun, e, epoch_steps = params["return_fun"], params["e"], params["epoch_steps"]
    window_size = e*epoch_steps if window_size is None else window_size
    ts = []
    for step_idx in range(len(self.measurements_flat)):
      
      at_the_time = self.measurements_flat[max(step_idx-window_size+1,0):step_idx+1]
      ts.append(return_fun(at_the_time))
    if chunked is False:
      return ts
    else:
      return list(chunks(ts, epoch_steps))

    
  def guess(self, epoch_steps, e, mode):
    if mode == "sum":
      return_fun = sum
    elif mode == "last":
      return_fun = lambda x: x[-1]
    elif mode == "first":
      return_fun = lambda x: x[0]
    elif mode == "fd":
      return_fun = lambda x: x[-1] - x[-2] if len(x) >= 2 else 0
    elif mode == "R":
      return_fun = lambda x: -(x[-1] - x[-2]) + x[0] if len(x) >= 2 else x[0]


    if self.epoch_steps is None:
      epoch_steps = len(self.measurements[0])
    else:
      epoch_steps = self.epoch_steps

    if e is None:
      e = self.e

    return {"e":e, "epoch_steps":epoch_steps, "return_fun":return_fun}

    
  def get(self, measurements_flat=None, e=None, mode=None):
    if mode is None:
      mode = self.mode

    params = self.guess(e=e, mode=mode, epoch_steps=None)
    return_fun, e, epoch_steps = params["return_fun"], params["e"], params["epoch_steps"]
    
    if measurements_flat is None:
      measurements_flat = self.measurements_flat

    desired_start = e*epoch_steps
    # return return_fun(self.measurements[metric][start_epoch:])
    return return_fun(measurements_flat[-desired_start:])
  
  def __repr__(self):
    return str(self.measurements)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def test_SumOfWhatever():
  x=SumOfWhatever()
  epochs = 3
  steps_per_epoch = 5
  returned_vals = []
  for i in range(epochs):
    for j in range(steps_per_epoch):
      x.update(i, j)
      returned_vals.append(x.get())
  assert returned_vals == [0, 1, 3, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

  x=SumOfWhatever()
  epochs = 3
  steps_per_epoch = 5
  returned_vals = []
  for i in range(epochs):
    for j in range(steps_per_epoch):
      x.update(i, j+i)
      returned_vals.append(x.get())
  assert returned_vals == [0, 1, 3, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    