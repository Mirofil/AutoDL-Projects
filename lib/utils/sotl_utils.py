import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
from pathlib import Path
import functools

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
import pickle
from tqdm import tqdm


def checkpoint_arch_perfs(archs, arch_metrics, epochs, steps_per_epoch, checkpoint_freq = None):
  """ (?) This appears to be a logging utility for the Seaborn chart but its mostly useless then I guess
  Outputs dict of shape {counter -> List of values (order unimportant)}
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

def arch_percentiles(arch_dict=None, percentiles = [0, 25, 50, 75, 100], mode="perf"):
  """Returns Dict[arch_str -> quartile_of_performance] """
  if arch_dict is None:
    arch_dict = load_arch_overview(mode = mode)
  arch_list = list(arch_dict.items()) # List of (arch_str, metric) tuples
  arch_list = sorted(arch_list, key=lambda x: x[1]) # Highest values are last
  percentiles_dict = {}
  for i in range(len(percentiles)-1):
    for arch_tuple in arch_list[round(len(arch_list)*(percentiles[i]/100)):round(len(arch_list)*(percentiles[i+1]/100))]:
      percentiles_dict[arch_tuple[0]] = percentiles[i+1]
  return percentiles_dict

def load_arch_overview(mode="perf"):
  """Load the precomputed performances of all architectures because querying NASBench is slow"""
  from pathlib import Path
  try:
    with open(f'./configs/nas-benchmark/percentiles/{mode}_all_dict.pkl', 'rb') as f:
      archs_dict = pickle.load(f)
    print(f"Suceeded in loading architectures from ./configs/nas-benchmark/percentiles/configs/nas-benchmark/percentiles/{mode}_all_dict.pkl! We have archs with len={len(archs_dict)}.")
    return archs_dict

  except Exception as e:
    print(f"Failed to load {mode} all dict! Need to run training with perf_percentile=0.9 to generate it. The error was {e}")
    raise NotImplementedError
  
def get_true_rankings(archs, api, hp='200', avg_all=False):
  """Extract true rankings of architectures on NASBench """
  final_accs = {genotype:summarize_results_by_dataset(genotype, api, separate_mean_std=False, avg_all=avg_all, hp=hp) for genotype in archs}
  true_rankings = {}
  for dataset in final_accs[archs[0]].keys():
    acc_on_dataset = [{"arch":arch.tostr(), "metric": final_accs[arch][dataset]} for i, arch in enumerate(archs)]
    acc_on_dataset = sorted(acc_on_dataset, key=lambda x: x["metric"], reverse=True)

    true_rankings[dataset] = acc_on_dataset
  
  return true_rankings, final_accs

def calc_corrs_val(archs, valid_accs, final_accs, true_rankings, corr_funs=None):
  if corr_funs is None:
    corr_funs = {"kendall": lambda x,y: scipy.stats.kendalltau(x,y).correlation, 
      "spearman":lambda x,y: scipy.stats.spearmanr(x,y).correlation, 
      "pearson":lambda x, y: scipy.stats.pearsonr(x,y)[0]}
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

def rank_inversions(combined_ranking, count_range=None):
  # x = [1,5,3,2,4]
  # y = [1,2,3,4,5]
  # rank_inversions(x,y,count_range=(5,5)) .. 1
  # rank_inversions(x,y,count_range=(4,5)) .. 1.5
  if count_range is None:
    count_range=(0, len(combined_ranking))
  elif count_range[0] > len(combined_ranking) or count_range[1] > len(combined_ranking):
    return 0

  sum_rank = 0
  count_rank = 0
  for idx in range(count_range[0]-1, count_range[1]):
    count_rank += 1
    sum_rank += abs(combined_ranking[idx][0] - combined_ranking[idx][1])

  return round(sum_rank/count_rank, 2)


def calc_corrs_after_dfs(epochs:int, xloader, steps_per_epoch:int, metrics_depth_dim, final_accs, archs, true_rankings, 
  prefix, api, corr_funs=None, wandb_log=False, corrs_freq=4, nth_tops=[1,5,10,20,30,40,50], constant=False, inversions=True):
  """Main function for producing correlation curves """
  # NOTE this function is useful for the sideffects of logging to WANDB
  # xloader should be the same dataLoader used to train since it is used here only for to reproduce indexes used in training. TODO we dont need both xloader and steps_per_epoch necessarily
  if corrs_freq is None:
    corrs_freq = 1
  if corr_funs is None:
    ranges_inversions = [(1,5),(1,10),(10, 50), (50, 90), (90,100), (1, 100), (30, 70)] if inversions else []
    funs_inversions = {f"inv{inversion_range[0]}to{inversion_range[1]}": lambda x, z=inversion_range: rank_inversions(x, z) for inversion_range in ranges_inversions}
    corr_funs = {"kendall": lambda x,y: scipy.stats.kendalltau(x,y).correlation, 
      "spearman":lambda x,y: scipy.stats.spearmanr(x,y).correlation, 
      "pearson":lambda x, y: scipy.stats.pearsonr(x,y)[0],
      **funs_inversions}

  sotl_rankings = []
  for epoch_idx in range(epochs):
    rankings_per_epoch = []
    for batch_idx, data in enumerate(xloader):
      if (steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx > steps_per_epoch:
        break
      if constant == True and batch_idx > 0:
        rankings_per_epoch.append(rankings_per_epoch[-1])
        continue
      relevant_sotls = []
      for i, arch in enumerate(metrics_depth_dim.keys()):
        try:
          if len(metrics_depth_dim[arch][epoch_idx]) - 1 >= batch_idx:
            metric = metrics_depth_dim[arch][epoch_idx][batch_idx]
          elif len(metrics_depth_dim[arch][epoch_idx]) > 0:
            metric = metrics_depth_dim[arch][epoch_idx][-1]
          else:
            metric = 3.14259
          relevant_sotls.append({"arch":arch, "metric": metrics_depth_dim[arch][epoch_idx][batch_idx]})
        except Exception as e:
          print(f"{e} for key={prefix}")
      #NOTE we need this sorting because we query the top1/top5 perf later down the line...
      relevant_sotls = sorted(relevant_sotls, key=lambda x: x["metric"], reverse=True) # This sorting takes 50% of total time - the code in the for loops takes miliseconds though it repeats a lot
      # vals = np.array([x["metric"] for x in relevant_sotls]) #TODO seems to be giving reverse order actually..
      # idxs = np.argpartition(vals, kth=min(5, len(vals)-1))
      # relevant_sotls = [relevant_sotls[idx] for idx in idxs]
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
      
      if constant == True and batch_idx > 0:
        #NOTE never occurs in first iteration of for loop
        to_log[epoch_idx].append(to_log[epoch_idx][-1])
        corrs_per_epoch.append(corr_per_dataset)
        continue

      corr_per_dataset = {}
      for dataset in final_accs[archs[0]].keys(): # the dict keys are all Dataset names
        ranking_pairs = [] # Ranking pairs do not necessarily have to be sorted. The scipy correlation routines sort it either way
        #NOTE true_rankings should be sorted already
        hash_index = {(str(true_ranking_dict["arch"]) if type(true_ranking_dict["arch"]) is str else true_ranking_dict["arch"].tostr()):true_ranking_dict['metric'] for pos, true_ranking_dict in enumerate(true_rankings[dataset])}
        for sotl_dict in [tuple2 for tuple2 in sotl_rankings[epoch_idx][batch_idx]]: #See the relevant_sotls instantiation 
          arch, sotl_metric = sotl_dict["arch"], sotl_dict["metric"]

          true_ranking_idx = hash_index[arch if type(arch) is str else arch.tostr()]
          ranking_pairs.append((sotl_metric, true_ranking_idx))

        ranking_pairs = np.array(ranking_pairs)
        approx_ranking = scipy.stats.rankdata(ranking_pairs[:, 0])

        if inversions:
          reference_ranking = scipy.stats.rankdata(ranking_pairs[:, 1])
          combined_ranking = sorted([(x,y) for x,y in zip(approx_ranking, reference_ranking)], key = lambda x: x[1])
          inversions_dict = {method: fun(combined_ranking) for method, fun in corr_funs.items() if "inv" in method}
        else:
          inversions = {}
        corr_per_dataset[dataset] = {**{method:fun(ranking_pairs[:, 0], ranking_pairs[:, 1]) for method, fun in corr_funs.items() if "inv" not in method}, **inversions_dict}

      top1_perf = summarize_results_by_dataset(sotl_rankings[epoch_idx][batch_idx][0]["arch"], api, separate_mean_std=False)
      top_perfs = {}
      bottom_perfs = {}
      for top in nth_tops:
        top_perf = {nth_top: summarize_results_by_dataset(sotl_rankings[epoch_idx][batch_idx][nth_top]["arch"], api, separate_mean_std=False) 
          for nth_top in range(min(5, len(sotl_rankings[epoch_idx][batch_idx])))}
        top_perf = avg_nested_dict(top_perf)
        top_perfs["top"+str(top)] = top_perf

        bottom_perf = {nth_top: summarize_results_by_dataset(sotl_rankings[epoch_idx][batch_idx][-nth_top]["arch"], api, separate_mean_std=False) 
          for nth_top in range(min(5, len(sotl_rankings[epoch_idx][batch_idx])))}
        bottom_perf = avg_nested_dict(bottom_perf)
        bottom_perfs["worst"+str(top)] = bottom_perf

      stats_to_log = {prefix:{**corr_per_dataset, "top1":top1_perf, **top_perfs, **bottom_perfs, "batch": batch_idx, "epoch":epoch_idx}, "true_step_corr":true_step}
      if wandb_log:
        wandb.log(stats_to_log)
      to_log[epoch_idx].append(stats_to_log)
      corrs_per_epoch.append(corr_per_dataset)
      
      true_step += corrs_freq

    corrs.append(corrs_per_epoch)
  
  return corrs, to_log

def grad_scale(parameters, norm: float, norm_type: float = 2.0) -> torch.Tensor:
  """Scales gradient in-place to have the desired total norm always"""
  with torch.no_grad():
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    max_norm = float(norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = norm/(total_norm + 1e-6)
    if norm is not None and norm > 0:
      for p in parameters:
          p.detach().mul_(clip_coef.to(p.device))
  return clip_coef, total_norm

class ValidAccEvaluator:
  def __init__(self, valid_loader, valid_loader_iter=None):
    self.valid_loader = valid_loader
    self.valid_loader_iter=valid_loader_iter
    super().__init__()

  def evaluate(self, arch, network, criterion, grads=False):
    network.eval()
    sampled_arch = arch
    with torch.set_grad_enabled(grads):
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

      if grads:
        loss.backward()


    network.train()
    return val_acc_top1, val_acc_top5, loss.item()

class DefaultDict_custom(dict):
  """
  default dict created by Teast Ares.
  """
  # def __init__(self, *args, default_item, **kwargs):
  #   super().__init__(*args,**kwargs)
  #   self.default_item = default_item
  def set_default_item(self, default_item):
      self.default_item = default_item
      
  def __missing__(self, key):
      x = deepcopy(self.default_item)
      self[key] = x
      return x

def estimate_grad_moments(xloader, network, criterion, steps=None):
  """Estimates mean/sd of gradients without any training steps inbetween - this gives the total_XXX kind of estimators which are only evaled once per epoch on the whole dataset"""
  with torch.set_grad_enabled(True):
    network.eval()
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      if steps is not None and step >= steps:
        break
      arch_targets = arch_targets.cuda(non_blocking=True)
      # prediction
      _, logits = network(arch_inputs.cuda(non_blocking=True))
      arch_loss = criterion(logits, arch_targets)
      arch_loss.backward()
    mean_grads = [p.grad.detach().clone() for p in network.parameters() if p.grad is not None]
    network.zero_grad()
    second_central_moment = []
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      if steps is not None and step >= steps:
        break
      arch_targets = arch_targets.cuda(non_blocking=True)
      # prediction
      _, logits = network(arch_inputs.cuda(non_blocking=True))
      arch_loss = criterion(logits, arch_targets)
      arch_loss.backward()
      for i, (g, mean_g) in enumerate(zip([p.grad.detach().clone() for p in network.parameters() if p.grad is not None], mean_grads)):

        if step == 0:
          second_central_moment.append(torch.pow(g-mean_g, 2))
        else:
          second_central_moment[i] = second_central_moment[i] + torch.pow(g-mean_g, 2)
      network.zero_grad()
    
    total_steps = len(xloader) if steps is None else steps
    for g in second_central_moment:
      g.multiply_(1/total_steps)

  network.train()
  network.zero_grad()
  return mean_grads, second_central_moment

def analyze_grads(network, grad_metrics: Dict, true_step=-1, arch_param_count=None, zero_grads=True, decay=0.995, total_steps=None, device='cuda'):
  """Computes gradient metrics for logging later. Works in-place in grad_metrics """

  with torch.no_grad():
    # TODO should try to explicitly exclude Arch parameters? Should not make a difference for SPOS regardless
    for k, log_k in [("grad_accum_tensor", "grad_accum"), ("grad_accum_singleE_tensor", "grad_accum_singleE"), ("grad_accum_decay_tensor", "grad_accum_decay")]:
      if grad_metrics[k] is not None and not (type(grad_metrics[k]) is int and grad_metrics[k] == 0):
        for g, dw in zip(grad_metrics[k], [p.grad.detach() for p in network.parameters() if p.grad is not None]):
          g.add_(dw)
      else:
        grad_metrics[k] = [p.grad.detach() for p in network.parameters() if p.grad is not None]
      if k != "grad_accum_decay":
        grad_metrics[log_k] = torch.sum(torch.stack([torch.norm(dp, 1) for dp in grad_metrics[k]])).item()
      else:
        grad_metrics[k] = [g*decay for g in grad_metrics[k]]
        grad_metrics[log_k] = torch.sum(torch.stack([torch.norm(dp, 1) for dp in grad_metrics[k]])).item()
    if grad_metrics["signs"] is None:
      grad_metrics["signs"] = [torch.sign(p.grad.detach()) for p in network.parameters() if p.grad is not None]
    else:
      for g, dw in zip(grad_metrics["signs"], [torch.sign(p.grad.detach()) for p in network.parameters() if p.grad is not None]):
        g.add_(dw)

    for k, log_k in [("grad_var_accum_tensor", "grad_var_accum"), ("grad_var_decay_accum_tensor", "grad_var_decay_accum")]:
      if grad_metrics[k] is None:
        grad_metrics[k] = [torch.zeros(p.size()).to('cuda') for p in network.parameters() if p.grad is not None]
      else:
        if "_decay" in k:
          grad_metrics[k] = [g*decay for g in grad_metrics[k]]
          mean_grads = [g/450 for g in grad_metrics["grad_accum_decay_tensor"]] # 450 is there since the weight of decay^450 is very low already so its a bit like 1 epoch worth of accum
        else:
          mean_grads = [g/total_steps for g in grad_metrics["grad_accum_tensor"]]
        for g, dw, mean_g in zip(grad_metrics[k], [p.grad.detach() for p in network.parameters() if p.grad is not None], mean_grads):
          g.add_(torch.pow(dw.to(device)-mean_g.to(device), 2))
      grad_metrics[log_k] = torch.sum(torch.stack([torch.norm(dp, 1) for dp in grad_metrics[k]])).item()

    grad_stack = torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in network.parameters() if p.grad is not None])
    grad_metrics["gn"] = torch.norm(grad_stack, 2).item() 
    grad_metrics["gnL1"] = torch.norm(grad_stack, 1).item() 
    for k in ["sogn", "sognL1"]:
      if grad_metrics[k] is None or k not in grad_metrics.keys():
        grad_metrics[k] = grad_metrics[k[2:]]
      else:
        grad_metrics["sogn"] += torch.norm(grad_stack, 2).item() 
        grad_metrics["sognL1"] += torch.norm(grad_stack, 1).item() 

    if arch_param_count is None: # Better to query NASBench API earlier I think
      arch_param_count = sum(p.numel() for p in network.parameters() if p.grad is not None) # p.requires_grad does not work here because the archiecture sampling is implemented by zeroing out some connections which makes the grads None, but they still have require_grad=True 
    grad_metrics["grad_normalized"] = grad_metrics["gn"] / arch_param_count
    grad_metrics["grad_mean_sign"] = torch.mean(torch.stack([g.mean() for g in grad_metrics["signs"]])/max(true_step, 1)).item()
    grad_metrics["grad_mean_accum"] = grad_metrics["grad_accum"]/(arch_param_count if arch_param_count is not None else -1)

  if zero_grads:
    network.zero_grad()
    for p in network.parameters():
      p.grad = None

def closest_epoch(api, arch_str, val, metric = "train-loss"):
  """NOTE val should be a metric such that lower is better!
  Also might be worthwhile to do binary search, but because we only ever train for short amount of times, all the hits should be very early on in the loop"""
  arch_idx = api.archstr2index[arch_str]
  found_change = False
  for i in range(199):
    info = api.get_more_info(arch_idx, "cifar10", iepoch=i, hp="200")
    next_info = api.get_more_info(arch_idx, "cifar10", iepoch=i+1, hp="200")
    if (val < info[metric] and val >= next_info[metric]):
      found_change = True
      break
  if found_change:
    return {"epoch":i, "val": info[metric]}
  elif val > next_info[metric]: # Worse loss than even one epoch of training
    return {"epoch": 0, "val": api.get_more_info(arch_idx, "cifar10", iepoch=0, hp="200")[metric]}
  elif val <= next_info[metric]: # Better loss than at the true end of training
    return {"epoch": 199, "val": api.get_more_info(arch_idx, "cifar10", iepoch=199, hp="200")[metric]}

def estimate_epoch_equivalents(archs: List, network, train_loader, criterion, api, steps=15) -> Dict:
  epoch_equivs = {}
  for arch in tqdm(archs, desc = "Estimating epochs equivalents"):
    network2 = deepcopy(network)
    network2.set_cal_mode('dynamic', arch)
    avg_loss = AverageMeter()
    with torch.no_grad():
      for batch_idx, data in tqdm(enumerate(train_loader), desc = "Iterating over batches", disable = True if len(train_loader) < 150000 else False):
        if batch_idx >= steps:
          break
        inputs, targets = data
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        _, logits = network2(inputs)
        loss = criterion(logits, targets)
        avg_loss.update(loss.item())
      epoch_equivs[arch.tostr()] = closest_epoch(api = api, arch_str = arch.tostr(), val = avg_loss.avg, metric='train-loss')

  return epoch_equivs

def init_grad_metrics(keys = ["train", "val", "total_train", "total_val"]):
  factory = DefaultDict_custom()
  factory.set_default_item(None)
  grad_metrics={k:factory for k in keys}
  return grad_metrics

def eval_archs_on_batch(xloader, archs, network, criterion, same_batch=False, metric="acc"):
  arch_metrics = []
  loader_iter = iter(xloader)
  inputs, targets = next(loader_iter)
  network.eval()
  with torch.no_grad():
    for i, sampled_arch in tqdm(enumerate(archs), desc = "Evaling archs on a batch of data"):
      network.set_cal_mode('dynamic', sampled_arch)
      if not same_batch:
        try:
          inputs, targets = next(loader_iter)
        except Exception as e:
          loader_iter = iter(xloader)
          inputs, targets = next(loader_iter)
      _, logits = network(inputs.cuda(non_blocking=True))
      loss = criterion(logits, targets.cuda(non_blocking=True))

      acc_top1, acc_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
      if metric == "acc":
        arch_metrics.append(acc_top1.item())
      elif metric == "loss":
        arch_metrics.append(-loss.item()) # Negative loss so that higher is better - as with validation accuracy
  network.train()
  return arch_metrics

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


def summarize_results_by_dataset(genotype: str = None, api=None, results_summary=None, separate_mean_std=False, avg_all=False, iepoch=None, hp = '200') -> dict:
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
  if not avg_all:
    for dataset in results_summary[0].keys():

      if separate_mean_std:
          interim[dataset]= {"mean":round(sum([result[dataset] for result in results_summary])/len(results_summary), 2),
          "std": round(np.std(np.array([result[dataset] for result in results_summary])), 2)}
      else:
          interim[dataset] = round(sum([result[dataset] for result in results_summary])/len(results_summary), 2)
  else:
    interim["avg"] = round(sum([result[dataset] for result in results_summary for dataset in results_summary[0].keys()])/len(results_summary[0].keys()), 2)
        
  return interim


class RecordedMetric:
  def __init__(self, name, e, return_fn):
    self.name = name
    self.e = e
    if return_fn == 'sum':
      self.return_fn = sum
    elif return_fn == "last":
      self.return_fn = lambda x: x[-1]
      
def rolling_window(a, window):
    if type(a) is list:
      a = np.array(a)
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
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
    