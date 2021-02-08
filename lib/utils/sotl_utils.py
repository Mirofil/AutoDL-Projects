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

def get_true_rankings(archs, api):
  final_accs = {genotype:summarize_results_by_dataset(genotype, api, separate_mean_std=False) for genotype in archs}
  true_rankings = {}
  for dataset in final_accs[archs[0]].keys():
    acc_on_dataset = [{"arch":arch, "acc": final_accs[arch][dataset]} for i, arch in enumerate(archs)]
    acc_on_dataset = sorted(acc_on_dataset, key=lambda x: x["acc"], reverse=True)

    true_rankings[dataset] = acc_on_dataset
  
  return true_rankings, final_accs

def calc_corrs_val(archs, valid_accs, final_accs, true_rankings, corr_funs):
  corr_per_dataset = {}
  for dataset in final_accs[archs[0]].keys():
    ranking_pairs = []
    for val_acc_ranking_idx, archs_idx in enumerate(np.argsort(-1*np.array(valid_accs))):
      arch = archs[archs_idx]
      for true_ranking_idx, arch2 in enumerate([tuple2_2[0] for tuple2_2 in true_rankings[dataset]]):
        if arch == arch2:
          ranking_pairs.append((val_acc_ranking_idx, true_ranking_idx))
          break
    ranking_pairs = np.array(ranking_pairs)
    corr_per_dataset[dataset] = {method:fun(ranking_pairs[:, 0], ranking_pairs[:, 1]) for method, fun in corr_funs.items()}
    
  return corr_per_dataset

def avg_nested_dict(d):
  # https://stackoverflow.com/questions/57311453/calculate-average-values-in-a-nested-dict-of-dicts
  try:
    d = list(d.values())
  except: 
    pass # we get into this branch on recursive calls
  _data = sorted([i for b in d for i in b.items()], key=lambda x:x[0])
  _d = [(a, [j for _, j in b]) for a, b in itertools.groupby(_data, key=lambda x:x[0])]
  return {a:avg_nested_dict(b) if isinstance(b[0], dict) else round(sum(b)/float(len(b)), 1) for a, b in _d}

def calc_corrs_after_dfs(epochs, xloader, steps_per_epoch, metrics_depth_dim, final_accs, archs, true_rankings, corr_funs, prefix, api):
  # NOTE this function is useful for the sideffects of logging to WANDB
  # xloader should be the same dataLoader used to train since it is used here only for to reproduce indexes used in training

  sotl_rankings = []
  for epoch_idx in range(epochs):
    rankings_per_epoch = []
    for batch_idx, data in enumerate(xloader):
      if (steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx > steps_per_epoch:
        break

      relevant_sotls = [{"arch": arch, "sotl": metrics_depth_dim[arch][epoch_idx][batch_idx]} for i, arch in enumerate(metrics_depth_dim.keys())]
      relevant_sotls = sorted(relevant_sotls, key=lambda x: x["sotl"], reverse=True) # This sorting takes 50% of total time - the code in the for loops takes miliseconds though it repeats a lot
      rankings_per_epoch.append(relevant_sotls)

    sotl_rankings.append(rankings_per_epoch)
   
  corrs = []
  true_step = 0
  for epoch_idx in range(epochs):
    corrs_per_epoch = []
    for batch_idx, data in enumerate(xloader):
      start = time.time()
      if (steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx > steps_per_epoch:
        break
      true_step += 1
      corr_per_dataset = {}
      for dataset in final_accs[archs[0]].keys(): # the dict keys are all Dataset names
        ranking_pairs = []

        hash_index = {str(arch["arch"]):pos for pos, arch in enumerate(true_rankings[dataset])}
        for sotl_ranking_idx, arch in enumerate([tuple2["arch"] for tuple2 in sotl_rankings[epoch_idx][batch_idx]]): #See the relevant_sotls instantiation 

          true_ranking_idx = hash_index[str(arch)]
          ranking_pairs.append((sotl_ranking_idx, true_ranking_idx))
          # for true_ranking_idx, arch2 in enumerate([tuple2_2["arch"] for tuple2_2 in true_rankings[dataset]]):
          #   if arch == arch2:
          #     ranking_pairs.append((sotl_ranking_idx, true_ranking_idx))
        ranking_pairs = np.array(ranking_pairs)
        corr_per_dataset[dataset] = {method:fun(ranking_pairs[:, 0], ranking_pairs[:, 1]) for method, fun in corr_funs.items()}
      start = time.time()
      top1_perf = summarize_results_by_dataset(sotl_rankings[epoch_idx][batch_idx][0]["arch"], api, separate_mean_std=False)
      top5 = {nth_top:summarize_results_by_dataset(sotl_rankings[epoch_idx][batch_idx][nth_top]["arch"], api, separate_mean_std=False) 
        for nth_top in range(min(5, len(sotl_rankings[epoch_idx][batch_idx])))}
      top5_perf = avg_nested_dict(top5)
      start = time.time()
      wandb.log({prefix:{**corr_per_dataset, "top1":top1_perf, "top5":top5_perf, "batch": batch_idx, "epoch":epoch_idx}, "true_step":true_step})
      corrs_per_epoch.append(corr_per_dataset)

    corrs.append(corrs_per_epoch)
  
  return corrs

def calculate_valid_acc_single_arch(valid_loader, arch, network, criterion):

  loader_iter = iter(valid_loader)
  network.eval()
  sampled_arch = arch
  with torch.no_grad():
    network.set_cal_mode('dynamic', sampled_arch)
    try:
      inputs, targets = next(loader_iter)
    except:
      loader_iter = iter(valid_loader)
      inputs, targets = next(loader_iter)
    _, logits = network(inputs.cuda(non_blocking=True))
    loss = criterion(logits, targets.cuda(non_blocking=True))
    val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
    val_acc_top1 = val_top1.item()
    val_acc_top5 = val_top5.item()

  network.train()
  return val_acc_top1, val_acc_top5, loss

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
  hp:str, account_time:bool=True, metric:str='valid-accuracy', e:int=1, is_random:bool=True):
  max_epoch = 199 if hp == '200' else 11

  observed_metrics, time_costs = [], []
  for epoch_idx in range(max_epoch):
    observed_metric, latency, time_cost, total_time_cost = simulate_train_eval_sotl(api=api, 
      arch=arch, dataset=dataset, hp=hp, iepoch=epoch_idx, account_time=account_time, 
      e=e, metric=metric, is_random=is_random)
    observed_metrics.append(observed_metric)
    time_costs.append(time_cost)



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

    # if dataset == "cifar10": # TODO I think this is not great in hindsight?
    #     dataset = "cifar10-valid"

    if e > 1:
        losses = []
        for i in range(iepoch - e + 1, iepoch + 1): # Sum up the train losses over multiple preceding epochs
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


def summarize_results_by_dataset(genotype: str = None, api=None, results_summary=None, separate_mean_std=False) -> dict:
  if results_summary is None:
    abridged_results = query_all_results_by_arch(genotype, api, iepoch=199, hp='200')
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


