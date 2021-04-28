##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v1 --rand_seed 777 --dry_run=True
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v1 --drop_path_rate 0.3
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v1
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v2 --rand_seed 777 
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v2
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v2
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo gdas --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo gdas
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo gdas
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo setn --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo setn
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo setn
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 51 --cand_eval_method sotl --steps_per_epoch 15 --train_batch_size 128 --eval_epochs 1 --eval_candidate_num 5 --val_batch_size 32 --scheduler constant --overwrite_additional_training True --dry_run=False --individual_logs False --greedynas_epochs=1
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch 10 --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 64 --dry_run=True --train_batch_size 64 --val_dset_ratio 0.2
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 3 --cand_eval_method sotl --steps_per_epoch None --eval_epochs 1
# python ./exps/NATS-algos/search-cell.py --algo=random --cand_eval_method=sotl --data_path=$TORCH_HOME/cifar.python --dataset=cifar10 --eval_epochs=2 --rand_seed=2 --steps_per_epoch=None
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo random
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch 5 --train_batch_size 128 --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 32 --scheduler cos_fast --lr 0.003 --overwrite_additional_training True --dry_run=False --reinitialize True --individual_logs False
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777

# python ./exps/NATS-algos/search-cell.py --dataset cifar5m  --data_path 'D:\' --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch 5 --train_batch_size 128 --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 32 --scheduler cos_fast --lr 0.003 --overwrite_additional_training True --dry_run=True --reinitialize True --individual_logs False --total_samples=600000
# python ./exps/NATS-algos/search-cell.py --dataset cifar5m  --data_path 'D:\' --algo darts-v1 --rand_seed 774 --dry_run=True --train_batch_size=2 --mmap r --total_samples=600000
# python ./exps/NATS-algos/search-cell.py --dataset cifar5m  --data_path '$TORCH_HOME/cifar.python' --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch 5 --train_batch_size 128 --eval_epochs 100 --eval_candidate_num 2 --val_batch_size 32 --scheduler cos_fast --lr 0.003 --overwrite_additional_training True --dry_run=True --reinitialize True --individual_logs False
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 3 --cand_eval_method sotl --steps_per_epoch 5 --train_batch_size 128 --eval_epochs 1 --eval_candidate_num 3 --val_batch_size 32 --scheduler cos_fast --lr 0.003 --overwrite_additional_training True --dry_run=True --reinitialize True --individual_logs False --resample=double_random
######################################################################################

import os, sys, time, random, argparse, math
import numpy as np
from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import count_parameters_in_MB, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nats_bench   import create
from utils.sotl_utils import (wandb_auth, query_all_results_by_arch, summarize_results_by_dataset,
  eval_archs_on_batch, 
  calc_corrs_after_dfs, calc_corrs_val, get_true_rankings, SumOfWhatever, checkpoint_arch_perfs, 
  ValidAccEvaluator, DefaultDict_custom, analyze_grads, estimate_grad_moments, grad_scale, 
  arch_percentiles, init_grad_metrics, closest_epoch, estimate_epoch_equivalents)
from models.cell_searchs.generic_model import ArchSampler
from log_utils import Logger

import wandb
import itertools
import scipy.stats
import time
import seaborn as sns
import bisect
sns.set_theme(style="whitegrid")

from argparse import Namespace
from typing import *
from tqdm import tqdm
import multiprocess as mp
from utils.wandb_utils import train_stats_reporter


# The following three functions are used for DARTS-V2
def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


def _hessian_vector_product(vector, network, criterion, base_inputs, base_targets, r=1e-2):
  R = r / _concat(vector).norm()
  for p, v in zip(network.weights, vector):
    p.data.add_(R, v)
  _, logits = network(base_inputs)
  loss = criterion(logits, base_targets)
  grads_p = torch.autograd.grad(loss, network.alphas)

  for p, v in zip(network.weights, vector):
    p.data.sub_(2*R, v)
  _, logits = network(base_inputs)
  loss = criterion(logits, base_targets)
  grads_n = torch.autograd.grad(loss, network.alphas)

  for p, v in zip(network.weights, vector):
    p.data.add_(R, v)
  return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


def backward_step_unrolled(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets, meta_learning=False):
  # _compute_unrolled_model
  if meta_learning in ['all', 'arch_only']:
    base_inputs = arch_inputs
    base_targets = arch_targets
  _, logits = network(base_inputs)
  loss = criterion(logits, base_targets)
  LR, WD, momentum = w_optimizer.param_groups[0]['lr'], w_optimizer.param_groups[0]['weight_decay'], w_optimizer.param_groups[0]['momentum']
  with torch.no_grad():
    theta = _concat(network.weights)
    try:
      moment = _concat(w_optimizer.state[v]['momentum_buffer'] for v in network.weights)
      moment = moment.mul_(momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, network.weights)) + WD*theta
    params = theta.sub(LR, moment+dtheta)
  unrolled_model = deepcopy(network)
  model_dict  = unrolled_model.state_dict()
  new_params, offset = {}, 0
  for k, v in network.named_parameters():
    if 'arch_parameters' in k: continue
    v_length = np.prod(v.size())
    new_params[k] = params[offset: offset+v_length].view(v.size())
    offset += v_length
  model_dict.update(new_params)
  unrolled_model.load_state_dict(model_dict)

  unrolled_model.zero_grad()
  _, unrolled_logits = unrolled_model(arch_inputs)
  unrolled_loss = criterion(unrolled_logits, arch_targets)
  unrolled_loss.backward()

  dalpha = unrolled_model.arch_parameters.grad
  vector = [v.grad.data for v in unrolled_model.weights]
  [implicit_grads] = _hessian_vector_product(vector, network, criterion, base_inputs, base_targets)
  
  dalpha.data.sub_(LR, implicit_grads.data)

  if network.arch_parameters.grad is None:
    network.arch_parameters.grad = deepcopy( dalpha )
  else:
    network.arch_parameters.grad.data.copy_( dalpha.data )
  return unrolled_loss.detach(), unrolled_logits.detach()

def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
  return {key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key]
          for key in state_dict_1.keys()}

def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, algo, logger, args=None, epoch=None, smoke_test=False, 
  meta_learning=False, api=None, supernets_decomposition=None, arch_groups_quartiles=None, arch_groups_brackets: Dict=[None], 
  all_archs=None, grad_metrics_percentiles=None, metrics_percs=None, percentiles=None, loss_threshold=None, replay_buffer = None):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  all_brackets = set(arch_groups_brackets.values())

  losses_percs = {"perc"+str(percentile): AverageMeter() for percentile in percentiles}
  supernet_train_stats = {"train_loss":{"sup"+str(percentile): [] for percentile in all_brackets}, 
    "val_loss": {"sup"+str(percentile): [] for percentile in all_brackets},
    "val_acc": {"sup"+str(percentile): [] for percentile in all_brackets},
    "train_acc": {"sup"+str(percentile): [] for percentile in all_brackets}}
  supernet_train_stats_avgmeters = {}
  for k in list(supernet_train_stats.keys()):
    supernet_train_stats[k+str("AVG")] = {"sup"+str(percentile): [] for percentile in all_brackets}
    supernet_train_stats_avgmeters[k+str("AVG")] = {"sup"+str(percentile): AverageMeter() for percentile in all_brackets}

  end = time.time()
  network.train()
  parsed_algo = algo.split("_")
  if (len(parsed_algo) == 3 and ("perf" in algo or "size" in algo)): # Can be used with algo=random_size_highest etc. so that it gets parsed correctly
    arch_sampler = ArchSampler(api=api, model=network, mode=parsed_algo[1], prefer=parsed_algo[2])
  else:
    arch_sampler = ArchSampler(api=api, model=network, mode="perf", prefer="random") # TODO mode=perf is a placeholder so that it loads the perf_all_dict, but then we do sample(mode=random) so it does not actually exploit the perf information

  grad_norm_meter = AverageMeter() # NOTE because its placed here, it means the average will restart after every epoch!
  if args.reptile is not None:
    model_init = deepcopy(network)
  arch_overview = {"cur_arch": None, "all_cur_archs": [], "all_archs": [], "top_archs_last_epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in tqdm(enumerate(xloader), desc = "Iterating over SearchDataset", total = len(xloader)): # Accumulate gradients over backward for sandwich rule
    if smoke_test and step >= 3:
      break
    if step == 0:
      logger.log(f"New epoch of arch; for debugging, those are the indexes of the first minibatch in epoch: {base_targets[0:10]}")
    scheduler.update(None, 1.0 * step / len(xloader))
    base_inputs = base_inputs.cuda(non_blocking=True)
    arch_inputs = arch_inputs.cuda(non_blocking=True)
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    sampling_done = False # Used for GreedyNAS online search space pruning - might have to resample many times until we find an architecture below the required threshold
    lowest_loss_arch = None
    lowest_loss = 10000
    if (args.sandwich is None or args.sandwich == 1):
      num_iters = 1
    else:
      num_iters = args.sandwich
    for outer_iter in range(num_iters):
      # Update the weights
      if args.reptile is None or (args.reptile is not None and step % args.reptile == 0): # For Reptile, we do not want to resample on every iteration
        if (args.reptile is not None and step % args.reptile == 0) or step == len(xloader):
          # Prepare for the interpolation step of Reptile
          new_state_dict = interpolate_state_dicts(model_init.state_dict(), network.state_dict(), args.reptile_weight)
          model_init = deepcopy(network)
          network.load_state_dict(new_state_dict)
        while not sampling_done: # TODO the sampling_done should be useful for like online sampling with rejections maybe
          if algo == 'setn':
            sampled_arch = network.dync_genotype(True)
            network.set_cal_mode('dynamic', sampled_arch)
          elif algo == 'gdas':
            network.set_cal_mode('gdas', None)
          elif algo.startswith('darts'):
            network.set_cal_mode('joint', None)
          elif "random_" in algo and len(parsed_algo) > 1 and ("perf" in algo or "size" in algo):
            sampled_arch = arch_sampler.sample()
            arch_overview["cur_arch"] = sampled_arch
            network.set_cal_mode('dynamic', sampled_arch)
          elif "random" in algo and args.sandwich is not None and args.sandwich > 1 and args.sandwich_mode == "quartiles":
            assert args.sandwich == 4 # 4 corresponds to using quartiles
            if step == 0:
              logger.log(f"Sampling from the Sandwich branch with sandwich={args.sandwich} and sandwich_mode={args.sandwich_mode}")
            sampled_archs = arch_sampler.sample(mode="quartiles") # Always samples 4 new archs but then we pick the one from the right quartile
            sampled_arch = sampled_archs[outer_iter]
            arch_overview["cur_arch"] = sampled_arch

            network.set_cal_mode('dynamic', sampled_arch)
          elif "random_" in algo and "grad" in algo:
            network.set_cal_mode('urs')
          elif algo == 'random': # NOTE the original branch needs to be last so that it is fall-through for all the special 'random' branches
            if supernets_decomposition or all_archs is not None or arch_groups_brackets is not None:
              if all_archs is not None:
                sampled_arch = random.sample(all_archs, 1)[0]
              else:
                sampled_arch = arch_sampler.sample(mode="random")
              arch_overview["cur_arch"] = sampled_arch
              network.set_cal_mode('dynamic', sampled_arch)
            else:
              network.set_cal_mode('urs', None)
          elif algo == 'enas':
            with torch.no_grad():
              network.controller.eval()
              _, _, sampled_arch = network.controller()
            network.set_cal_mode('dynamic', sampled_arch)
          else:
            raise ValueError('Invalid algo name : {:}'.format(algo))
          if loss_threshold is not None:
            with torch.no_grad():
              _, logits = network(base_inputs)
              base_loss = criterion(logits, base_targets) * (1 if args.sandwich is None else 1/args.sandwich)
              if base_loss.item() < lowest_loss:
                lowest_loss = base_loss.item()
                lowest_loss_arch = sampled_arch
              if base_loss.item() < loss_threshold:
                sampling_done = True
          else:
            sampling_done = True

          if sampling_done:
            arch_overview["cur_arch"] = sampled_arch
            arch_overview["all_archs"].append(sampled_arch)
            arch_overview["all_cur_archs"].append(sampled_arch)
        
      network.zero_grad()
      _, logits = network(base_inputs)
      base_loss = criterion(logits, base_targets) * (1 if args.sandwich is None else 1/args.sandwich)

      if outer_iter == num_iters - 1 and replay_buffer is not None and args.replay_buffer > 0: # We should only do the replay once regardless of the architecture batch size
        for replay_arch in replay_buffer:

          network.set_cal_mode('dynamic', replay_arch)
          _, logits = network(base_inputs)
          replay_loss = criterion(logits, base_targets)
          if epoch in [0,1] and step == 0:
            logger.log(f"Replay loss={replay_loss.item()} for {len(replay_buffer)} items with num_iters={num_iters}, outer_iter={outer_iter}, replay_buffer={replay_buffer}") # Debugging messages

          base_loss = base_loss + (args.replay_buffer_weight / args.replay_buffer) * replay_loss # TODO should we also specifically add the L2 regularizations as separate items? Like this, it diminishes the importance of weight decay here
          network.set_cal_mode('dynamic', arch_overview["cur_arch"])

      base_loss.backward()

      if 'gradnorm' in algo: # Normalize gradnorm so that all updates have the same norm. But does not work well at all in practice
        # tn = torch.norm(torch.stack([torch.norm(p.detach(), 2).to('cuda') for p in w_optimizer.param_groups[0]["params"]]), 2)
        # print(f"TOtal norm before  before {tn}")
        coef, total_norm = grad_scale(w_optimizer.param_groups[0]["params"], grad_norm_meter.avg)
        grad_norm_meter.update(total_norm)
        # tn = torch.norm(torch.stack([torch.norm(p.detach(), 2).to('cuda') for p in w_optimizer.param_groups[0]["params"]]), 2)    
        # print(f"TOtal norm before  after {tn}")
      if supernets_decomposition is not None:
        # TODO need to fix the logging here I think. The normal logging is much better now
        cur_quartile = arch_groups_quartiles[sampled_arch.tostr()]
        with torch.no_grad():
          dw = [p.grad.detach().to('cpu') if p.grad is not None else torch.zeros_like(p).to('cpu') for p in network.parameters()]
          cur_supernet = supernets_decomposition[cur_quartile]
          for decomp_w, g in zip(cur_supernet.parameters(), dw):
            if decomp_w.grad is not None:
              decomp_w.grad.copy_(g)
            else:
              decomp_w.grad = g
          analyze_grads(cur_supernet, grad_metrics_percentiles["perc"+str(cur_quartile)]["supernet"], true_step =step+epoch*len(xloader), total_steps=step+epoch*len(xloader))
        
        if type(arch_groups_quartiles) is dict:
          for quartile in arch_groups_quartiles.keys():
            if quartile == cur_quartile:
              losses_percs["perc"+str(quartile)].update(base_loss.item()) # TODO this doesnt make any sense

      base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
      base_losses.update(base_loss.item() / (1 if args.sandwich is None else 1/args.sandwich),  base_inputs.size(0))
      base_top1.update  (base_prec1.item(), base_inputs.size(0))
      base_top5.update  (base_prec5.item(), base_inputs.size(0))
      arch_overview["train_acc"].append(base_prec1)
      arch_overview["train_loss"].append(base_loss.item())

      cur_bracket = arch_groups_brackets[arch_overview["cur_arch"].tostr()]
      if type(arch_groups_brackets) is dict:
        for key, val in [("train_loss", base_loss.item() / (1 if args.sandwich is None else 1/args.sandwich)), ("train_acc", base_prec1.item())]:
          for bracket in all_brackets:
            if bracket == cur_bracket:
              supernet_train_stats[key]["sup"+str(cur_bracket)].append(val)
              supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(cur_bracket)].update(val)
              supernet_train_stats[key+"AVG"]["sup"+str(cur_bracket)].append(supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(cur_bracket)].avg)

            else:
              item_to_add = supernet_train_stats[key]["sup"+str(bracket)][-1] if len(supernet_train_stats[key]["sup"+str(bracket)]) > 0 else 3.14159
            
              supernet_train_stats[key]["sup"+str(bracket)].append(item_to_add)
              avg_to_add = supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(bracket)].avg if supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(bracket)].avg > 0 else None
              supernet_train_stats[key+"AVG"]["sup"+str(bracket)].append(avg_to_add)

    w_optimizer.step()

    for previously_sampled_arch in arch_overview["all_cur_archs"]:
      # update the architecture-weight
      if algo == 'setn':
        network.set_cal_mode('joint')
      elif algo == 'gdas':
        network.set_cal_mode('gdas', None)
      elif algo.startswith('darts'):
        network.set_cal_mode('joint', None)
      elif 'random' in algo and len(arch_overview["all_cur_archs"]) > 1 and args.replay_buffer is not None:
        network.set_cal_mode('dynamic', previously_sampled_arch)
      elif 'random' in algo:
        network.set_cal_mode('urs', None)
      elif algo != 'enas':
        raise ValueError('Invalid algo name : {:}'.format(algo))
      network.zero_grad()
      if algo == 'darts-v2':
        arch_loss, logits = backward_step_unrolled(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets, meta_learning=meta_learning)
        a_optimizer.step()
      elif algo == 'random' or algo == 'enas' or 'random' in algo:
        with torch.no_grad():
          _, logits = network(arch_inputs)
          arch_loss = criterion(logits, arch_targets)
      else:
        _, logits = network(arch_inputs)
        arch_loss = criterion(logits, arch_targets)
        arch_loss.backward()
        a_optimizer.step()
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
      arch_overview["val_acc"].append(arch_prec1)
      arch_overview["val_loss"].append(arch_loss.item())

      cur_bracket = arch_groups_brackets[arch_overview["cur_arch"].tostr()]
      if type(arch_groups_brackets) is dict:
        for key, val in [("val_loss", arch_loss.item()), ("val_acc", arch_prec1.item())]:
          for bracket in all_brackets:
            if bracket == cur_bracket:
              supernet_train_stats[key]["sup"+str(bracket)].append(val)
              supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(bracket)].update(val)
              supernet_train_stats[key+"AVG"]["sup"+str(bracket)].append(supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(bracket)].avg)
            else:
              item_to_add = supernet_train_stats[key]["sup"+str(bracket)][-1] if len(supernet_train_stats[key]["sup"+str(bracket)]) > 0 else 3.14159
              
              supernet_train_stats[key]["sup"+str(bracket)].append(item_to_add)
              supernet_train_stats[key+"AVG"]["sup"+str(bracket)].append(supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(bracket)].avg)
    arch_overview["all_cur_archs"] = [] #Cleanup

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)

  print(f"Average gradient norm over last epoch was {grad_norm_meter.avg}, min={grad_norm_meter.min}, max={grad_norm_meter.max}")
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg, supernet_train_stats, arch_overview


def train_controller(xloader, network, criterion, optimizer, prev_baseline, epoch_str, print_freq, logger):
  # config. (containing some necessary arg)
  #   baseline: The baseline score (i.e. average val_acc) from the previous epoch
  data_time, batch_time = AverageMeter(), AverageMeter()
  GradnormMeter, LossMeter, ValAccMeter, EntropyMeter, BaselineMeter, RewardMeter, xend = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), time.time()
  
  controller_num_aggregate = 20
  controller_train_steps = 50
  controller_bl_dec = 0.99
  controller_entropy_weight = 0.0001

  network.eval()
  network.controller.train()
  network.controller.zero_grad()
  loader_iter = iter(xloader)
  for step in range(controller_train_steps * controller_num_aggregate):
    try:
      inputs, targets = next(loader_iter)
    except:
      loader_iter = iter(xloader)
      inputs, targets = next(loader_iter)
    inputs  = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - xend)
    
    log_prob, entropy, sampled_arch = network.controller()
    with torch.no_grad():
      network.set_cal_mode('dynamic', sampled_arch)
      _, logits = network(inputs)
      val_top1, val_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
      val_top1  = val_top1.view(-1) / 100
    reward = val_top1 + controller_entropy_weight * entropy
    if prev_baseline is None:
      baseline = val_top1
    else:
      baseline = prev_baseline - (1 - controller_bl_dec) * (prev_baseline - reward)
   
    loss = -1 * log_prob * (reward - baseline)
    
    # account
    RewardMeter.update(reward.item())
    BaselineMeter.update(baseline.item())
    ValAccMeter.update(val_top1.item()*100)
    LossMeter.update(loss.item())
    EntropyMeter.update(entropy.item())
  
    # Average gradient over controller_num_aggregate samples
    loss = loss / controller_num_aggregate
    loss.backward(retain_graph=True)

    # measure elapsed time
    batch_time.update(time.time() - xend)
    xend = time.time()
    if (step+1) % controller_num_aggregate == 0:
      grad_norm = torch.nn.utils.clip_grad_norm_(network.controller.parameters(), 5.0)
      GradnormMeter.update(grad_norm)
      optimizer.step()
      network.controller.zero_grad()

    if step % print_freq == 0:
      Sstr = '*Train-Controller* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, controller_train_steps * controller_num_aggregate)
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = '[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Reward {reward.val:.2f} ({reward.avg:.2f})] Baseline {basel.val:.2f} ({basel.avg:.2f})'.format(loss=LossMeter, top1=ValAccMeter, reward=RewardMeter, basel=BaselineMeter)
      Estr = 'Entropy={:.4f} ({:.4f})'.format(EntropyMeter.val, EntropyMeter.avg)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Estr)

  return LossMeter.avg, ValAccMeter.avg, BaselineMeter.avg, RewardMeter.avg


def get_best_arch(train_loader, valid_loader, network, n_samples, algo, logger, criterion,
  additional_training=True, api=None, style:str='val_acc', w_optimizer=None, w_scheduler=None, 
  config: Dict=None, epochs:int=1, steps_per_epoch:int=100, 
  val_loss_freq:int=1, train_stats_freq=3, overwrite_additional_training:bool=False, 
  scheduler_type:str=None, xargs:Namespace=None, train_loader_stats=None, val_loader_stats=None, model_config=None, all_archs=None):
  with torch.no_grad():
    network.eval()
    if 'random' in algo:
      if xargs.evenly_split is not None:
        arch_sampler = ArchSampler(api=api, model=network, mode=xargs.evenly_split)
        archs = arch_sampler.sample(mode="evenly_split", candidate_num=xargs.eval_candidate_num)
        decision_metrics = []
      elif api is not None and xargs is not None:
        archs, decision_metrics = network.return_topK(n_samples, True, api=api, dataset=xargs.dataset, size_percentile=xargs.size_percentile, perf_percentile=xargs.perf_percentile), []
      else:
        archs, decision_metrics = network.return_topK(n_samples, True), []
    elif algo == 'setn':
      archs, decision_metrics = network.return_topK(n_samples, False), []
    elif algo.startswith('darts') or algo == 'gdas':
      arch = network.genotype
      archs, decision_metrics = [arch], []
    elif algo == 'enas':
      archs, decision_metrics = [], []
      for _ in range(n_samples):
        _, _, sampled_arch = network.controller()
        archs.append(sampled_arch)
    else:
      raise ValueError('Invalid algorithm name : {:}'.format(algo))
    
    if all_archs is not None: # Overwrite the just sampled archs with the ones that were supplied. Useful in order to match up with the archs used in search_func
      logger.log(f"Overwrote arch sampling in get_best_arch with a subset of len={len(all_archs)}")
      archs = all_archs
    else:
      logger.log(f"Were not supplied any limiting subset of archs so instead just sampled fresh ones with len={len(archs)} using algo={algo}")
    logger.log(f"Running get_best_arch (evenly_split={xargs.evenly_split}, style={style}) with initial seeding of archs:{[api.archstr2index[arch.tostr()] for arch in archs[0:25]]}")
    
    # The true rankings are used to calculate correlations later
    true_rankings, final_accs = get_true_rankings(archs, api)
    upper_bound = {}
    for n in [1,5,10]:
      upper_bound[f"top{n}"] = {"cifar10":0, "cifar10-valid":0, "cifar100":0, "ImageNet16-120":0}
      for dataset in true_rankings.keys():
        upper_bound[f"top{n}"][dataset] += sum([x["metric"] for x in true_rankings[dataset][0:n]])/min(n, len(true_rankings[dataset][0:n]))
        # upper_bound["top1"][dataset] += sum([x["metric"] for x in true_rankings[dataset][0:1]])/1
    upper_bound = {"upper":upper_bound}
    logger.log(f"Upper bound: {upper_bound}")
    
    if steps_per_epoch is not None and steps_per_epoch != "None":
      steps_per_epoch = int(steps_per_epoch)
    elif steps_per_epoch in [None, "None"]:
      steps_per_epoch = len(train_loader)
    else:
      raise NotImplementedError

    if style in ['val_acc', 'val']:
      # Original code branch from the AutoDL repo, although slightly groomed
      if len(archs) > 1:
        decision_metrics = eval_archs_on_batch(xloader=valid_loader, archs=archs, network=network, criterion=criterion)
        corr_per_dataset = calc_corrs_val(archs=archs, valid_accs=decision_metrics, final_accs=final_accs, true_rankings=true_rankings, corr_funs=None)
        # wandb.log({"notrain_val":corr_per_dataset})
      else:
        decision_metrics=eval_archs_on_batch(xloader=valid_loader, archs=archs, network=network)

  if style == 'sotl' or style == "sovl":    
    if xargs.postnet_switch_train_val:
      logger.log("Switching train and validation sets for postnet training")
      train_loader, valid_loader = valid_loader, train_loader
    # Simulate short training rollout to compute SoTL for candidate architectures
    cond = logger.path('corr_metrics').exists() and not overwrite_additional_training
    total_metrics_keys = ["total_val", "total_train", "total_val_loss", "total_train_loss", "total_arch_count"]
    so_metrics_keys = ["sotl", "sovl", "sovalacc", "sotrainacc", "sovalacc_top5", "sogn", "sogn_norm"]
    grad_metric_keys = ["gn", "grad_normalized", "grad_mean_accum", "grad_accum", "grad_mean_sign"]
    pct_metric_keys = ["train_loss_pct"]
    metrics_keys = ["val_acc", "train_acc", "train_loss", "val_loss", "gap_loss", *pct_metric_keys, *grad_metric_keys, *so_metrics_keys, *total_metrics_keys]
    must_restart = False
    start_arch_idx = 0

    if cond: # Try to load previous checkpoint. It will restart if significant changes are detected in the current run from the checkpoint 
             # (this prevents accidentally using checkpoints for different params than the current ones)
      logger.log("=> loading checkpoint of the last-checkpoint '{:}' start".format(logger.path('corr_metrics')))

      checkpoint = torch.load(logger.path('corr_metrics'))
      checkpoint_config = checkpoint["config"] if "config" in checkpoint.keys() else {}

      try:
        # if type(list(checkpoint["metrics"]["sotl"].keys())[0]) is not str or type(checkpoint["metrics"]) is dict:
        #   must_restart = True # will need to restart metrics because using the old checkpoint format
        #   print("Using old checkpoint format! Must restart")
        metrics = {k:checkpoint["metrics"][k] for k in checkpoint["metrics"].keys()}
        train_stats = checkpoint["train_stats"]
        # prototype = metrics[metrics_keys[0]]
        # first_arch = next(iter(metrics[metrics_keys[0]].keys()))
        # if type(first_arch) is not str:
        #   first_arch = first_arch.tostr()
        # wrong_count=0
        # for metric_key in metrics_keys:
        #   if not (len(metrics[metric_key]) == len(prototype) and len(metrics[metric_key][first_arch]) == len(prototype[first_arch])):
        #     print(f"Found wrong metric length! For metric {metric_key}, we have {len(metrics[metric_key])} and {len(prototype)}, then also {len(metrics[metric_key][first_arch])} and {len(prototype[first_arch])}")
        #     print(f"Example 1: {str(metrics[metric_key])[0:300]}")
        #     print(f"Example 2: {str(prototype)[0:300]}")
        #     wrong_count +=1
        # if wrong_count >= len(metrics_keys):
        #   print("Must restart because lengths are wrong")
        # must_restart = True
      except Exception as e:
        print("Errored due to exception below")
        print(e)
        print("Unknown reason but must restart!")
        must_restart = True

      decision_metrics = checkpoint["decision_metrics"] if "decision_metrics" in checkpoint.keys() else []
      start_arch_idx = checkpoint["start_arch_idx"]
      cond1={k:v for k,v in checkpoint_config.items() if ('path' not in k and 'dir' not in k and k not in ["dry_run", "workers", "mmap"])}
      cond2={k:v for k,v in vars(xargs).items() if ('path' not in k and 'dir' not in k and k not in ["dry_run", "workers", "mmap"])}
      logger.log(f"Checkpoint config: {cond1}")
      logger.log(f"Newly input config: {cond2}")
      different_items = {k: cond1[k] for k in cond1 if k in cond2 and cond1[k] != cond2[k]}
      if (cond1 == cond2 or len(different_items) == 0):
        logger.log("Both configs are equal.")
      else:
        logger.log("Checkpoint and current config are not the same! need to restart")
        logger.log(f"Different items are : {different_items}")
      
      if set([x.tostr() if type(x) is not str else x for x in checkpoint["archs"]]) != set([x.tostr() if type(x) is not str else x for x in archs]):
        print("Checkpoint has sampled different archs than the current seed! Need to restart")
        print(f"Checkpoint: {checkpoint['archs'][0]}")
        print(f"Current archs: {archs[0]}")
        if all_archs is not None:
          logger.log("Architectures do not match up to the checkpoint but since all_archs was supplied, it might be intended")
        # must_restart = True
        else:
          logger.log("Using the checkpoint archs as ground-truth for current run. But might be better to investigate what went wrong")
          archs = checkpoint["archs"]
          true_rankings, final_accs = get_true_rankings(archs, api)
          upper_bound = {}
          for n in [1,5,10]:
            upper_bound[f"top{n}"] = {"cifar10":0, "cifar10-valid":0, "cifar100":0, "ImageNet16-120":0}
            for dataset in true_rankings.keys():
              upper_bound[f"top{n}"][dataset] += sum([x["metric"] for x in true_rankings[dataset][0:n]])/min(n, len(true_rankings[dataset][0:n]))
              # upper_bound["top1"][dataset] += sum([x["metric"] for x in true_rankings[dataset][0:1]])/1
          upper_bound = {"upper":upper_bound}

    if xargs.restart:
      must_restart=True
    if (not cond) or must_restart or (xargs is None) or (cond1 != cond2 and len(different_items) > 0): #config should be an ArgParse Namespace
      if not cond:
        logger.log(f"Did not find a checkpoint for supernet post-training at {logger.path('corr_metrics')}")
      else:
        logger.log(f"Starting postnet training with fresh metrics")

      metrics_factory = {arch.tostr():[[] for _ in range(epochs)] for arch in archs}
      # metrics = {k:{arch.tostr():[[] for _ in range(epochs)] for arch in archs} for k in metrics_keys}   
      metrics = DefaultDict_custom()
      metrics.set_default_item(metrics_factory)
      decision_metrics = []    
      start_arch_idx = 0

      train_stats = [[] for _ in range(epochs*steps_per_epoch+1)]

    train_start_time = time.time()
    arch_rankings = sorted([(arch.tostr(), summarize_results_by_dataset(genotype=arch, api=api, avg_all=True)["avg"]) for arch in archs], reverse=True, key=lambda x: x[1])
    # The arch_rankings_dict includes exact rank whereas arch_percentiles() returns brackets
    arch_rankings_dict = {k: {"rank":rank, "metric":v} for rank, (k,v) in enumerate(arch_rankings)}
    arch_rankings_thresholds = [min(math.floor(len(archs)*(threshold/100)), len(archs)-1) for threshold in [10,20,30,40,50,60,70,80,90,100]] # List where each entry is the real rank position of a percentile
    arch_rankings_thresholds_nominal = {real: nominal for real, nominal in zip(arch_rankings_thresholds, [10,20,30,40,50,60,70,80,90,100])} # Maps the real rank to percentile-based rank
    logger.log(f"Arch ranking threshols mapping from real-to-percentiles: {arch_rankings_thresholds_nominal}")

    if xargs.evenify_training:
      # Those two lines are just to get the proper criterion to use
      config_opt = load_config('./configs/nas-benchmark/hyper-opts/200E.config', None, logger)
      _, _, criterion = get_optim_scheduler(network.weights, config_opt)
      
      epoch_eqs = estimate_epoch_equivalents(archs=archs, network=network, api=api, criterion=criterion, train_loader=train_loader, steps=15)
      max_epoch_attained = max([x["val"] for x in epoch_eqs.values()])
      logger.log(f"Evenifying the training so that all architectures have the equivalent of {max_epoch_attained} of training measured by their own training curves")

    if xargs.adaptive_lr:
      lr_counts = defaultdict(int)

    for arch_idx, sampled_arch in tqdm(enumerate(archs[start_arch_idx:], start_arch_idx), desc="Iterating over sampled architectures", total = n_samples-start_arch_idx):
      assert (all_archs is None) or (sampled_arch in all_archs), "There must be a bug since we are training an architecture that is not in the supplied subset"
      arch_natsbench_idx = api.query_index_by_arch(sampled_arch)
      true_perf = summarize_results_by_dataset(sampled_arch, api, separate_mean_std=False)
      true_step = 0 # Used for logging per-iteration statistics in WANDB
      arch_str = sampled_arch.tostr() # We must use architectures converted to str for good serialization to pickle

      arch_threshold = arch_rankings_thresholds_nominal[arch_rankings_thresholds[bisect.bisect_left(arch_rankings_thresholds, arch_rankings_dict[sampled_arch.tostr()]["rank"])]]

      if xargs.resample not in [False, None, "False", "false", "None"]:
        assert xargs.reinitialize
        search_model = get_cell_based_tiny_net(model_config)
        search_model.set_algo(xargs.algo)
        network2 = search_model.to('cuda')
        network2.set_cal_mode('dynamic', sampled_arch)
        print(f"Reinitalized new supernetwork! First param weights sample: {str(next(iter(network2.parameters())))[0:100]}")
        if xargs.resample == "double_random":
          seed = random.choice(range(50))
          new_logger = Logger(xargs.save_dir, seed)
          last_info = new_logger.path('info')
          if last_info.exists(): # automatically resume from previous checkpoint
            logger.log("During double random sampling - loading checkpoint of the last-info '{:}' start".format(last_info))
            if os.name == 'nt': # The last-info pickles have PosixPaths serialized in them, hence they cannot be instantied on Windows
              import pathlib
              temp = pathlib.PosixPath
              pathlib.PosixPath = pathlib.WindowsPath
            last_info   = torch.load(last_info.resolve())
            checkpoint  = torch.load(last_info['last_checkpoint'])
            print(f"Sampled new supernetwork! First param weights sample before: {str(next(iter(network2.parameters())))[0:100]}")
            network2.load_state_dict( checkpoint['search_model'] )
            print(f"Sampled new supernetwork! First param weights sample after: {str(next(iter(network2.parameters())))[0:100]}")
          else:
            print(f"Couldnt find pretrained supernetwork for seed {seed} at {last_info}")
      else:
        network2 = deepcopy(network)
        network2.set_cal_mode('dynamic', sampled_arch)

      arch_param_count = api.get_cost_info(api.query_index_by_arch(sampled_arch), xargs.dataset if xargs.dataset != "cifar5m" else "cifar10")['params'] # we will need to do a forward pass to get the true count because of the superneetwork subsampling
      print(f"Arch param count: {arch_param_count}MB")

      if hasattr(train_loader.sampler, "reset_counter"):
        train_loader.sampler.reset_counter()

      if xargs.lr is not None and scheduler_type is None:
        scheduler_type = "constant"

      if xargs.adaptive_lr:
        assert xargs.scheduler == "constant"
        assert xargs.lr is None
        assert xargs.deterministic_loader in ["train", "all"] # Not strictly necessary but this assures tha the LR search uses the same data across all LR options

        if xargs.adaptive_lr == "1cycle":

          from torch_lr_finder import LRFinder
          network3 = deepcopy(network2)
          network3.logits_only = True
          w_optimizer3, _, criterion = get_optim_scheduler(network3.weights, config, attach_scheduler=False)

          lr_finder = LRFinder(network3, w_optimizer3, criterion, device="cuda")
          lr_finder.range_test(train_loader, start_lr=0.0001, end_lr=1, num_iter=100)
          best_lr = lr_finder.history["lr"][(np.gradient(np.array(lr_finder.history["loss"]))).argmin()]
          try:
            lr_plot_ax, weird_lr = lr_finder.plot(suggest_lr=True) # to inspect the loss-learning rate graph
          except:
            lr_plot_ax = lr_finder.plot(suggest_lr=False)
          lr_finder.reset() # to reset the model and optimizer to their initial state
          wandb.log({"lr_plot": lr_plot_ax}, commit=False)
        elif xargs.adaptive_lr == "custom":
          lrs = np.geomspace(1, 0.001, 10)
          lr_results = {}
          avg_of_avg_loss = AverageMeter()
          for lr in tqdm(lrs, desc="Searching LRs"):
            network3 = deepcopy(network2)
            print(str(list(network3.parameters()))[0:100])

            config = config._replace(scheduler='constant', constant_lr=lr)
            w_optimizer3, _, criterion = get_optim_scheduler(network3.weights, config)
            avg_loss = AverageMeter()
            for batch_idx, data in tqdm(enumerate(train_loader), desc = f"Training in order to find the best LR for arch_idx={arch_idx}", disable=True):
              if batch_idx > 20:
                break
              network3.zero_grad()
              inputs, targets = data
              inputs = inputs.cuda(non_blocking=True)
              targets = targets.cuda(non_blocking=True)
              _, logits = network3(inputs)
              train_acc_top1, train_acc_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
              loss = criterion(logits, targets)
              avg_loss.update(loss.item())
              loss.backward()
              w_optimizer3.step()
            lr_results[lr] = avg_loss.avg
            avg_of_avg_loss.update(avg_loss.avg)
          best_lr = min(lr_results, key = lambda k: lr_results[k])
          logger.log(lr_results)
          lr_counts[best_lr] += 1

        if arch_idx == 0:
          logger.log(f"Find best LR for arch_idx={arch_idx} at LR={best_lr}")

      logger.log(f"Picking the scheduler with scheduler_type={scheduler_type}")
      if scheduler_type in ['linear_warmup', 'linear']:
        config = config._replace(scheduler=scheduler_type, warmup=1, eta_min=0)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type == "cos_reinit":
        # In practice, this leads to constant LR = 0.025 since the original Cosine LR is annealed over 100 epochs and our training schedule is very short
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type in ['cos_adjusted']:
        config = config._replace(scheduler='cos', warmup=0, epochs=epochs)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type in ['cos_fast']:
        config = config._replace(scheduler='cos', warmup=0, LR=0.001 if xargs.lr is None else xargs.lr, epochs=epochs, eta_min=0)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type in ['cos_warmup']:
        config = config._replace(scheduler='cos', warmup=1, LR=0.001 if xargs.lr is None else xargs.lr, epochs=epochs, eta_min=0)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type in ["scratch"]:
        config_opt = load_config('./configs/nas-benchmark/hyper-opts/200E.config', None, logger)
        config_opt = config_opt._replace(LR=0.1 if xargs.lr is None else xargs.lr)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config_opt)
      elif scheduler_type in ["scratch12E"]:
        config_opt = load_config('./configs/nas-benchmark/hyper-opts/12E.config', None, logger)
        config_opt = config_opt._replace(LR=0.1 if xargs.lr is None else xargs.lr)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config_opt)
      elif scheduler_type in ["scratch1E"]:
        config_opt = load_config('./configs/nas-benchmark/hyper-opts/01E.config', None, logger)
        config_opt = config_opt._replace(LR=0.1 if xargs.lr is None else xargs.lr)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config_opt)
      elif (xargs.lr is not None or (xargs.lr is None and bool(xargs.adaptive_lr) == True)) and scheduler_type == 'constant':
        config = config._replace(scheduler='constant', constant_lr=xargs.lr if not xargs.adaptive_lr else best_lr)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      else:
        # NOTE in practice, since the Search function uses Cosine LR with T_max that finishes at end of search_func training, this switches to a constant 1e-3 LR.
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
        w_optimizer2.load_state_dict(w_optimizer.state_dict())
        w_scheduler2.load_state_dict(w_scheduler.state_dict())
      if arch_idx == start_arch_idx: #Should only print it once at the start of training
        logger.log(f"Optimizers for the supernet post-training: {w_optimizer2}, {w_scheduler2}")

      running = defaultdict(int)

      grad_metrics = init_grad_metrics(keys = ["train", "val", "total_train", "total_val"])

      start = time.time()
      val_loss_total, val_acc_total, _ = valid_func(xloader=val_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
      if xargs.grads_analysis:
        analyze_grads(network=network2, grad_metrics=grad_metrics["total_val"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)
      train_loss_total, train_acc_total, _ = valid_func(xloader=train_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
      if xargs.grads_analysis:
        analyze_grads(network=network2, grad_metrics=grad_metrics["total_train"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)
      val_loss_total, train_loss_total = -val_loss_total, -train_loss_total

      grad_mean, grad_std = estimate_grad_moments(xloader=train_loader, network=network2, criterion=criterion, steps=25)
      grad_std_scalar = torch.mean(torch.cat([g.view(-1) for g in grad_std], dim=0)).item()
      grad_snr_scalar = (grad_std_scalar**2)/torch.mean(torch.pow(torch.cat([g.view(-1) for g in grad_mean], dim=0), 2)).item()
      network2.zero_grad()

      if arch_idx == 0: # Dont need to print this for every architecture I guess
        logger.log(f"Time taken to compute total_train/total_val statistics once with {xargs.total_estimator_steps} estimator steps is {time.time()-start}")

      if xargs.individual_logs: # Log the training stats for each sampled architecture separately
        q = mp.Queue()
        # This reporting process is necessary due to WANDB technical difficulties. It is used to continuously report train stats from a separate process
        # Otherwise, when a Run is intiated from a Sweep, it is not necessary to log the results to separate training runs. But that it is what we want for the individual arch stats
        p=mp.Process(target=train_stats_reporter, kwargs=dict(queue=q, config=vars(xargs),
            sweep_group=f"Search_Cell_{algo}_arch", sweep_run_name=wandb.run.name or wandb.run.id or "unknown", sweep_id = wandb.run.sweep_id or "unknown", arch=sampled_arch.tostr()))
        p.start()

      if xargs.evenify_training:
        # Train each architecture until they all reach the same amount of training as a preprocessing step before recording the training statistics for correlations
        cur_epoch, target_loss = epoch_eqs[sampled_arch.tostr()]["epoch"], epoch_eqs[sampled_arch.tostr()]["val"]
        done = False
        iter_count=0
        while not done:
          avg_loss = AverageMeter()
          for batch_idx, data in tqdm(enumerate(train_loader), desc = f"Evenifying training for arch_idx={arch_idx}"):
            if avg_loss.avg < target_loss and batch_idx >= 15 and avg_loss.avg != 0:
              done = True
              break
            network2.zero_grad()
            inputs, targets = data
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            _, logits = network2(inputs)
            train_acc_top1, train_acc_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            loss = criterion(logits, targets)
            avg_loss.update(loss.item())

            loss.backward()
            w_optimizer2.step()
            iter_count += 1
        if arch_idx == 0:
          logger.log(f"Trained arch_idx={arch_idx} for {iter_count} iterations to make it match up to {max_epoch_attained}")
    
      for epoch_idx in range(epochs):
        if epoch_idx < 5:
          logger.log(f"New epoch of arch; for debugging, those are the indexes of the first minibatch in epoch with idx up to 5: {epoch_idx}: {next(iter(train_loader))[1][0:15]}")
          logger.log(f"Weights LR before scheduler update: {w_scheduler2.get_lr()[0]}")

        if epoch_idx == 0: # Here we construct the almost constant total_XXX metric time series (they only change once per epoch)
          total_mult_coef = min(len(train_loader)-1, steps_per_epoch)
        else:
          total_mult_coef = min(len(train_loader)-1, steps_per_epoch)

        for metric, metric_val in zip(["total_val", "total_train", "total_val_loss", "total_train_loss", "total_arch_count", "total_gstd", "total_gsnr"], [val_acc_total, train_acc_total, val_loss_total, train_loss_total, arch_param_count, grad_std_scalar, grad_snr_scalar]):
          metrics[metric][arch_str][epoch_idx] = [metric_val]*total_mult_coef

        val_acc_evaluator = ValidAccEvaluator(valid_loader, None)

        for batch_idx, data in tqdm(enumerate(train_loader), desc = "Iterating over batches", disable = True if len(train_loader) < 150000 else False):
          if (steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx > steps_per_epoch:
            break
          with torch.set_grad_enabled(mode=additional_training):
            if scheduler_type in ["linear", "linear_warmup"]:
              w_scheduler2.update(epoch_idx, 1.0 * batch_idx / min(len(train_loader), steps_per_epoch))
            elif scheduler_type == "cos_adjusted":
              w_scheduler2.update(epoch_idx , batch_idx/min(len(train_loader), steps_per_epoch))
            elif scheduler_type == "cos_reinit":
              w_scheduler2.update(epoch_idx, 0.0)
            elif scheduler_type in ['cos_fast', 'cos_warmup']:
              w_scheduler2.update(epoch_idx , batch_idx/min(len(train_loader), steps_per_epoch))
            else:
              w_scheduler2.update(epoch_idx, 1.0 * batch_idx / len(train_loader))

            network2.zero_grad()
            inputs, targets = data
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            _, logits = network2(inputs)
            train_acc_top1, train_acc_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            loss = criterion(logits, targets)
            if additional_training:
              loss.backward()
              w_optimizer2.step()
              analyze_grads(network=network2, grad_metrics=grad_metrics["train"], true_step=true_step, arch_param_count=arch_param_count, total_steps=true_step)
            loss, train_acc_top1, train_acc_top5 = loss.item(), train_acc_top1.item(), train_acc_top5.item()
            
          true_step += 1

          if batch_idx % val_loss_freq == 0:
            w_optimizer2.zero_grad() # NOTE We MUST zero gradients both before and after doing the fake val gradient calculations
            valid_acc, valid_acc_top5, valid_loss = val_acc_evaluator.evaluate(arch=sampled_arch, network=network2, criterion=criterion, grads=xargs.grads_analysis)
            if xargs.grads_analysis:
              analyze_grads(network=network2, grad_metrics=grad_metrics["val"], true_step=true_step, arch_param_count=arch_param_count, total_steps=true_step)
            w_optimizer2.zero_grad() # NOTE We MUST zero gradients both before and after doing the fake val gradient calculations

          running["sovl"] -= valid_loss
          running["sovalacc"] += valid_acc
          running["sovalacc_top5"] += valid_acc_top5
          running["sotl"] -= loss # Need to have negative loss so that the ordering is consistent with val acc
          running["sotrainacc"] += train_acc_top1
          running["sotrainacc_top5"] += train_acc_top5
          running["sogn"] += grad_metrics["train"]["sogn"]
          running["sogn_norm"] += grad_metrics["train"]["grad_normalized"]

          for k in [key for key in metrics_keys if key.startswith("so")]:
            metrics[k][arch_str][epoch_idx].append(running[k])
          metrics["val_acc"][arch_str][epoch_idx].append(valid_acc)
          metrics["train_acc"][arch_str][epoch_idx].append(train_acc_top1)
          metrics["train_loss"][arch_str][epoch_idx].append(-loss)
          metrics["val_loss"][arch_str][epoch_idx].append(-valid_loss)
          metrics["gap_loss"][arch_str][epoch_idx].append(-valid_loss + (loss - valid_loss))
          if xargs.adaptive_lr == "custom":
            metrics["lr_avg_loss"][arch_str][epoch_idx].append(-avg_of_avg_loss.avg)

          if len(metrics["train_loss"][arch_str][epoch_idx]) >= 3:
            loss_normalizer = sum(metrics["train_loss"][arch_str][epoch_idx][-3:])/3
          elif epoch_idx >= 1:
            loss_normalizer = sum(metrics["train_loss"][arch_str][epoch_idx-1][-3:])/3
          else:
            loss_normalizer = 1
          metrics["train_loss_pct"][arch_str][epoch_idx].append(1-loss/loss_normalizer)
          data_types = ["train"] if not xargs.grads_analysis else ["train", "val", "total_train", "total_val"]
          grad_log_keys = ["gn", "gnL1", "sogn", "sognL1", "grad_normalized", "grad_accum", "grad_accum_singleE", "grad_accum_decay", "grad_mean_accum", "grad_mean_sign", "grad_var_accum", "grad_var_decay_accum"]

          for data_type in data_types:
            for log_key in grad_log_keys:
              val = grad_metrics[data_type][log_key]
              metrics[data_type+"_"+log_key][arch_str][epoch_idx].append(grad_metrics[data_type][log_key])

          if xargs.grads_analysis:
            metrics["gap_grad_accum"][arch_str][epoch_idx].append(metrics["train_grad_accum"][arch_str][epoch_idx][-1]-metrics["val_grad_accum"][arch_str][epoch_idx][-1])

          special_metrics = {k:metrics[k][arch_str][epoch_idx][-1] for k in metrics.keys() if len(metrics[k][arch_str][epoch_idx])>0}
          special_metrics = {**special_metrics, **{k+str(arch_threshold):v for k,v in special_metrics.items()}}
          batch_train_stats = {"lr":w_scheduler2.get_lr()[0], f"lr{arch_threshold}":w_scheduler2.get_lr()[0],
           "true_step":true_step, "train_loss":loss, f"train_loss{arch_threshold}":loss, 
          f"epoch_eq{arch_threshold}": closest_epoch(api, arch_str, loss, metric="train-loss")["epoch"],
            "train_acc_top1":train_acc_top1, f"train_acc_top1{arch_threshold}":train_acc_top1, "train_acc_top5":train_acc_top5, 
            "valid_loss":valid_loss, f"valid_loss{arch_threshold}":valid_loss, "valid_acc":valid_acc, f"valid_acc{arch_threshold}":valid_acc,
            "valid_acc_top5":valid_acc_top5, 
            "grad_train":grad_metrics["train"]["gn"], f"grad_train{arch_threshold}":grad_metrics["train"]["gn"],
            "train_epoch":epoch_idx, "train_batch":batch_idx, **special_metrics, 
            "true_perf":true_perf, f"true_perf{arch_threshold}":true_perf,
            "arch_param_count":arch_param_count, f"arch_param_count{arch_threshold}":arch_param_count, "arch_idx": arch_natsbench_idx, 
            "arch_rank":arch_threshold}

          train_stats[epoch_idx*steps_per_epoch+batch_idx].append(batch_train_stats)
          if xargs.individual_logs and true_step % train_stats_freq == 0:
            q.put(batch_train_stats)

        if additional_training:
          val_loss_total, val_acc_total, _ = valid_func(xloader=val_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
          if xargs.grads_analysis:
            analyze_grads(network=network2, grad_metrics=grad_metrics["total_val"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)
          network2.zero_grad()
          train_loss_total, train_acc_total, _ = valid_func(xloader=train_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
          if xargs.grads_analysis:
            analyze_grads(network=network2, grad_metrics=grad_metrics["total_train"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)   
          val_loss_total, train_loss_total = -val_loss_total, -train_loss_total
          network2.zero_grad()
          grad_mean, grad_std = estimate_grad_moments(xloader=train_loader, network=network2, criterion=criterion, steps=25)
          grad_std_scalar = torch.mean(torch.cat([g.view(-1) for g in grad_std], dim=0)).item()
          grad_snr_scalar = (grad_std_scalar**2)/torch.mean(torch.pow(torch.cat([g.view(-1) for g in grad_mean], dim=0), 2)).item()
          network2.zero_grad()
        for metric, metric_val in zip(["total_val", "total_train", "total_val_loss", "total_train_loss", "total_arch_count", "total_gstd", "total_gsnr"], [val_acc_total, train_acc_total, val_loss_total, train_loss_total, arch_param_count, grad_std_scalar, grad_snr_scalar]):
          metrics[metric][arch_str][epoch_idx].append(metric_val)

        #Cleanup at end of epoch
        grad_metrics["train"]["grad_accum_singleE"] = None
        grad_metrics["val"]["grad_accum_singleE"] = None
        if hasattr(train_loader.sampler, "reset_counter"):
          train_loader.sampler.counter += 1

      final_metric = None # Those final/decision metrics are not very useful apart from being a compatibility layer with how get_best_arch worked in the base repo
      if style == "sotl":
        final_metric = running["sotl"]
      elif style == "sovl":
        final_metric = running["sovl"]

      decision_metrics.append(final_metric)

      corr_metrics_path = save_checkpoint({"corrs":{}, "metrics":metrics, "train_stats":train_stats,
        "archs":archs, "start_arch_idx": arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},   
        logger.path('corr_metrics'), logger, quiet=True)

      if xargs.individual_logs:
        q.put("SENTINEL") # This lets the Reporter process know it should quit
            
    if xargs.adaptive_lr:
      logger.log(f"Distribution of LRs from adaptive LR search is {lr_counts}")

    train_total_time = time.time()-train_start_time
    print(f"Train total time: {train_total_time}")

    wandb.run.summary["train_total_time"] = train_total_time

    original_metrics = deepcopy(metrics)

    print(list(metrics["train_loss"].keys()))
    metrics_FD = {k+"FD": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=1).get_time_series(chunked=True, mode="fd") for arch in archs} for k,v in metrics.items() if k in ['val_acc', 'train_loss', 'val_loss']}
    metrics.update(metrics_FD)

    if epochs > 1:
      metrics_E1 = {k+"E1": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=1).get_time_series(chunked=True) for arch in archs} for k,v in metrics.items() if not k.startswith("so") and not 'accum' in k and not 'total' in k}
      metrics.update(metrics_E1)

      Einf_metrics = ["train_lossFD", "train_loss_pct"]
      metrics_Einf = {k+"Einf": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=100).get_time_series(chunked=True) for arch in archs} for k,v in metrics.items() if k in Einf_metrics and not k.startswith("so") and not 'accum' in k and not 'total' in k}
      metrics.update(metrics_Einf)      
    else:
      # We only calculate Sum-of-FD metrics in this case
      metrics_E1 = {k+"E1": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=1).get_time_series(chunked=True) for arch in archs} for k,v in metrics.items() if "FD" in k }
      metrics.update(metrics_E1)
    for key in metrics_FD.keys(): # Remove the pure FD metrics because they are useless anyways
      metrics.pop(key, None)

    start=time.time()
    corrs = {}
    to_logs = []

    for k,v in tqdm(metrics.items(), desc="Calculating correlations"):
      if torch.is_tensor(v[next(iter(v.keys()))]):
        v = {inner_k: [[batch_elem.item() for batch_elem in epoch_list] for epoch_list in inner_v] for inner_k, inner_v in v.items()}
      # We cannot do logging synchronously with training becuase we need to know the results of all archs for i-th epoch before we can log correlations for that epoch
      constant_metric = True if (k in total_metrics_keys or "upper" in k) else False
      corr, to_log = calc_corrs_after_dfs(epochs=epochs, xloader=train_loader, steps_per_epoch=steps_per_epoch, metrics_depth_dim=v, 
    final_accs = final_accs, archs=archs, true_rankings = true_rankings, prefix=k, api=api, wandb_log=False, corrs_freq = xargs.corrs_freq, constant=constant_metric)
      corrs["corrs_"+k] = corr
      to_logs.append(to_log)

    arch_ranking_inner = [{"arch":arch, "metric":metrics["total_arch_count"][arch][0][0]} for arch in metrics["total_arch_count"].keys()]
    arch_ranking_inner = sorted(arch_ranking_inner, key=lambda x: x["metric"], reverse=True)
    arch_true_rankings = {"cifar10":arch_ranking_inner, "cifar100":arch_ranking_inner,"cifar10-valid":arch_ranking_inner, "ImageNet16-120":arch_ranking_inner}
    for k in ["train_grad_accum", "train_lossE1", "sotl", "train_grad_mean_accum", "sogn"]:
      if k not in metrics.keys():
        print(f"WARNING! Didnt find {k} in metrics keys: {list(metrics.keys())}")
        continue
      v = metrics[k]
      if torch.is_tensor(v[next(iter(v.keys()))]):
        v = {inner_k: [[batch_elem.item() for batch_elem in epoch_list] for epoch_list in inner_v] for inner_k, inner_v in v.items()}
      corr, to_log = calc_corrs_after_dfs(epochs=epochs, xloader=train_loader, steps_per_epoch=steps_per_epoch, metrics_depth_dim=v, 
    final_accs = final_accs, archs=archs, true_rankings = arch_true_rankings, corr_funs=None, prefix=k+"P", api=api, wandb_log=False, corrs_freq = xargs.corrs_freq, constant=None)
      corrs["param_corrs_"+k] = corr
      to_logs.append(to_log) 

    print(f"Calc corrs time: {time.time()-start}")
    arch_perf_tables = {}
    arch_perf_charts = {}
    for metric in ['val_acc', 'train_loss']:
      arch_perf_checkpoints = checkpoint_arch_perfs(archs=archs, arch_metrics=metrics[metric], epochs=epochs, 
        steps_per_epoch = len(to_logs[0][0]), checkpoint_freq=None)
      interim_arch_perf = []
      for key in arch_perf_checkpoints.keys():
        transformed = [(key, value) for value in arch_perf_checkpoints[key]]
        interim_arch_perf.extend(transformed)
      # interim_arch_perf looks like [(1,5), (1, 6.25), (1,4.75), (5,8), (5,9), (5,12)]
      interim_arch_perf = np.array(interim_arch_perf)
      arch_perf_table = wandb.Table(data = interim_arch_perf, columns=["iter", "perf"])
      arch_perf_chart = sns.swarmplot(x = interim_arch_perf[:, 0], y = interim_arch_perf[:, 1])
      f = arch_perf_chart.get_figure()
      f.suptitle(f"Sampled arch performances at checkpoints using the metric: {metric}")
      f.axes[0].set_xlabel("Checkpointed iteration (in minibatches)")
      f.axes[0].set_ylabel("Validation accuracy")

      arch_perf_charts[metric] = f
      arch_perf_tables[metric] = arch_perf_table

    if n_samples-start_arch_idx > 0: #If there was training happening - might not be the case if we just loaded checkpoint
      # We reshape the stored train statistics so that it is a Seq[Dict[k: summary statistics across all archs for a timestep]] instead of Seq[Seq[Dict[k: train stat for a single arch]]]
      processed_train_stats = []
      all_threshold_keys = {}
      for key in batch_train_stats.keys():
        all_threshold_keys[key] = None
        for threshold in arch_rankings_thresholds_nominal.values():
          all_threshold_keys[key+str(threshold)] = None

      # all_threshold_keys = {f"train_loss{x}":None for x in arch_rankings_thresholds} # TODO I think this does nothing? It was in the 'for k, v in {**batch, **all_loss}



      for idx, stats_across_time in tqdm(enumerate(train_stats), desc="Processing train stats"):
        filtered_out = {}
        agg = {k: np.array([single_train_stats[k] if k in single_train_stats.keys() else np.nan for single_train_stats in stats_across_time]) for k, v in all_threshold_keys.items()}
        for k, v in agg.items():
          if issubclass(type(v), dict) or (type(v) is list and type(v[0]) is dict):
            continue
          else:
            try:
              np.all(np.isnan(v))
            except:
              continue
            if not np.all(np.isnan(v)):
              filtered_out[k] = v
        agg = {k: {"mean":np.nanmean(v), "std": np.nanstd(v), "min":np.nanmin(v), "max":np.nanmax(v)} for k,v in filtered_out.items()}
        agg["true_step"] = idx
        processed_train_stats.append(agg)

    ### Here, we aggregate various local variables containing logging-worthy objects and put them all together for wandb.log to work nicely. NOTE I didnt know about wandb.log(commit=False) at the time
    for epoch_idx in range(len(to_logs[0])):
      relevant_epochs = [to_logs[i][epoch_idx] for i in range(len(to_logs))]
      for batch_idx in range(len(relevant_epochs[0])):
        relevant_batches = [relevant_epoch[batch_idx] for relevant_epoch in relevant_epochs]
        all_batch_data = {}
        for batch in relevant_batches:
          all_batch_data.update(batch)

        # Here we log both the aggregated train statistics and the correlations
        if n_samples-start_arch_idx > 0: #If there was training happening - might not be the case if we just loaded checkpoint
          all_data_to_log = {**all_batch_data, **{"summary":processed_train_stats[epoch_idx*steps_per_epoch+batch_idx]}} # We add the Summary nesting to prevent overwriting of the normal stats by the batch_train mean/stds
        else:
          all_data_to_log = all_batch_data

        all_data_to_log.update(upper_bound)

        wandb.log(all_data_to_log)

    wandb.log({"arch_perf":arch_perf_tables, "arch_perf_charts":arch_perf_charts})

  if style in ["sotl", "sovl"] and n_samples-start_arch_idx > 0: # otherwise, we are just reloading the previous checkpoint so should not save again
    corr_metrics_path = save_checkpoint({"metrics":original_metrics, "corrs": corrs, "train_stats": train_stats,
      "archs":archs, "start_arch_idx":arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},
      logger.path('corr_metrics'), logger)

    print(f"Upload to WANDB at {corr_metrics_path.absolute()}")
    try:
      wandb.save(str(corr_metrics_path.absolute()))
    except Exception as e:
      print(f"Upload to WANDB failed because {e}")

  best_idx = np.argmax(decision_metrics)
  try:
    best_arch, best_valid_acc = archs[best_idx], decision_metrics[best_idx]
  except:
    logger.log("Failed to get best arch via decision_metrics")
    logger.log(f"Decision metrics: {decision_metrics}")
    logger.log(f"Best idx: {best_idx}, length of archs: {len(archs)}")
    best_arch,best_valid_acc = archs[0], decision_metrics[0]
  return best_arch, best_valid_acc



def valid_func(xloader, network, criterion, algo, logger, steps=None, grads=False):
  data_time, batch_time = AverageMeter(), AverageMeter()
  loss, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  with torch.set_grad_enabled(grads):
    network.eval()
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      if steps is not None and step >= steps:
        break
      arch_targets = arch_targets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - end)
      # prediction
      _, logits = network(arch_inputs.cuda(non_blocking=True))
      arch_loss = criterion(logits, arch_targets)
      if grads:
        arch_loss.backward()
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      loss.update(arch_loss.item(),  arch_inputs.size(0))
      top1.update  (arch_prec1.item(), arch_inputs.size(0))
      top5.update  (arch_prec5.item(), arch_inputs.size(0))
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
    

  network.train()
  return loss.avg, top1.avg, top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( max(int(xargs.workers), 1))
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(xargs)

  if xargs.dataset_postnet is None:
    xargs.dataset_postnet = xargs.dataset

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1, mmap=xargs.mmap, total_samples=xargs.total_samples)
  if xargs.overwite_epochs is None:
    extra_info = {'class_num': class_num, 'xshape': xshape}
  else:
    extra_info = {'class_num': class_num, 'xshape': xshape, 'epochs': xargs.overwite_epochs}
  config = load_config(xargs.config_path, extra_info, logger)
  if xargs.search_epochs is not None:
    config = config._replace(epochs=xargs.search_epochs)
  resolved_train_batch_size, resolved_val_batch_size = xargs.train_batch_size if xargs.train_batch_size is not None else config.batch_size, xargs.val_batch_size if xargs.val_batch_size is not None else config.test_batch_size
  # NOTE probably better idea to not use train_batch_size here to not accidentally change the supernet search?
  search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', 
    (config.batch_size, config.test_batch_size), workers=xargs.workers, epochs=config.epochs + config.warmup, determinism=xargs.deterministic_loader, 
    merge_train_val = xargs.merge_train_val_supernet, merge_train_val_and_use_test = xargs.merge_train_val_and_use_test)
  logger.log(f"Using train batch size: {resolved_train_batch_size}, val batch size: {resolved_val_batch_size}")
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces(xargs.search_space, 'nats-bench')

  model_config = dict2config(
      dict(name='generic', C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes, num_classes=class_num,
           space=search_space, affine=bool(xargs.affine), track_running_stats=bool(xargs.track_running_stats)), None)
  logger.log('search space : {:}'.format(search_space))
  logger.log('model config : {:}'.format(model_config))
  search_model = get_cell_based_tiny_net(model_config)
  search_model.set_algo(xargs.algo)
  # TODO this logging search model makes a big mess in the logs! And it is almost always the same anyways
  # logger.log('{:}'.format(search_model))

  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.weights, config)
  a_optimizer = torch.optim.Adam(search_model.alphas, lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay, eps=xargs.arch_eps)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  params = count_parameters_in_MB(search_model)
  logger.log('The parameters of the search model = {:.2f} MB'.format(params))
  logger.log('search-space : {:}'.format(search_space))
  if bool(xargs.use_api):
    api = create(None, 'topology', fast_mode=True, verbose=False)
  else:
    api = None
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  network, criterion = search_model.cuda(), criterion.cuda()  # use a single GPU
  last_info_orig, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')

  if last_info_orig.exists() and not xargs.reinitialize and not xargs.force_rewrite: # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info_orig))
    if os.name == 'nt': # The last-info pickles have PosixPaths serialized in them, hence they cannot be instantied on Windows
      import pathlib
      temp = pathlib.PosixPath
      pathlib.PosixPath = pathlib.WindowsPath
    last_info   = torch.load(last_info_orig.resolve())
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    genotypes   = checkpoint['genotypes']
    baseline  = checkpoint['baseline']
    valid_accuracies = checkpoint['valid_accuracies']
    search_model.load_state_dict( checkpoint['search_model'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    print(last_info_orig)
    logger.log("=> do not find the last-info file : {:}".format(last_info_orig))
    start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: network.return_topK(1, True)[0]}
    baseline = None
  
  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup if xargs.search_epochs is None else xargs.search_epochs
  # We simulate reinitialization by not training (+ not loading the saved state_dict earlier)
  if start_epoch > total_epoch: # In case we train for 500 epochs but then the default value for search epochs is only 100
    start_epoch = total_epoch
  if start_epoch == total_epoch - 1 and xargs.greedynas_epochs is not None and xargs.greedynas_epochs > 0:
    # Need to restart the LR schedulers
    logger = prepare_logger(xargs, path_suffix="greedy")
    logger.log(f"Start of GreedyNAS training at epoch={start_epoch}! Will train for {xargs.greedynas_epochs} epochs more.")
    config_greedynas = deepcopy(config)._replace(LR = xargs.greedynas_lr, epochs = xargs.greedynas_epochs)
    w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.weights, config_greedynas)

    last_info_orig, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
    if last_info_orig.exists() and not xargs.reinitialize and not xargs.force_rewrite: # automatically resume from previous checkpoint
      logger.log("Need to reload checkpoint due to using extra supernet training")
      logger.log("=> loading extra checkpoint of the last-info '{:}' start".format(last_info_orig))
      if os.name == 'nt': # The last-info pickles have PosixPaths serialized in them, hence they cannot be instantied on Windows
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
      last_info   = torch.load(last_info_orig.resolve())
      start_epoch = last_info['epoch']
      checkpoint  = torch.load(last_info['last_checkpoint'])
      genotypes   = checkpoint['genotypes']
      baseline  = checkpoint['baseline']
      valid_accuracies = checkpoint['valid_accuracies']
      search_model.load_state_dict( checkpoint['search_model'] )
      w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
      w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
      a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
      logger.log("=> loading extra checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))

  arch_groups_brackets =  arch_percentiles(percentiles=[0,10,20,30,40,50,60,70,80,90,100], mode="perf")
  if xargs.supernets_decomposition:
    percentiles = [0, 25, 50, 75, 100]
    empty_network = deepcopy(network).to('cpu') # TODO dont actually need to use those networks in the end? Can just use grad_metrics I think
    with torch.no_grad():
      for p in empty_network.parameters():
        p.multiply_(0.)
    supernets_decomposition = {percentiles[i+1]:empty_network for i in range(len(percentiles)-1)}
    supernets_decomposition["init"] = deepcopy(network)
    logger.log(f'Initialized {len(percentiles)} supernets because supernet_decomposition={xargs.supernets_decomposition}')
    arch_groups_quartiles = arch_percentiles(percentiles=percentiles, mode=xargs.supernets_decomposition_mode)
    if (last_info_orig.exists() and "grad_metrics_percs" not in checkpoint.keys()) or not last_info_orig.exists():
      # TODO what is the point of this archs_subset?
      archs_subset = network.return_topK(-1 if xargs.supernets_decomposition_topk is None else xargs.supernets_decomposition_topk, use_random=False) # Should return all archs for negative K
      grad_metrics_percs = {"perc"+str(percentiles[i+1]):init_grad_metrics(keys=["supernet"]) for i in range(len(percentiles)-1)}
    else:
      grad_metrics_percs = checkpoint["grad_metrics_percs"]
      archs_subset = checkpoint["archs_subset"]

    metrics_factory = {"perc"+str(percentile):[[] for _ in range(total_epoch)] for percentile in percentiles}
    metrics_percs = DefaultDict_custom()
    metrics_percs.set_default_item(metrics_factory)
    logger.log(f"Using all_archs (len={len(archs_subset)}) for modified algo=random sampling in order to execute the supernet decomposition")
  else:
    supernets_decomposition, arch_groups_quartiles, archs_subset, grad_metrics_percs, percentiles, metrics_percs = None, None, None, None, [None], None
  if xargs.greedynas_epochs is not None and xargs.greedynas_epochs > 0: # TODO should make it exploit the warmup supernet training?
    greedynas_archs = network.return_topK(xargs.eval_candidate_num, use_random=True)
    logger.log(f"Sampling architectures that will be used for GreedyNAS Supernet post-main-supernet training in search_func, head = {[api.archstr2index[x.tostr()] for x in greedynas_archs[0:10]]}")

  else:
    greedynas_archs = None
  supernet_key = "supernet"
  arch_perf_percs = {k:None for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
  replay_buffer = None
  for epoch in range(start_epoch if not xargs.reinitialize else 0, total_epoch + (xargs.greedynas_epochs if xargs.greedynas_epochs is not None else 0) if not xargs.reinitialize else 0):
    if epoch >= 3 and xargs.dry_run:
      print("Breaking training loop early due to smoke testing")
      break


    w_scheduler.update(epoch if epoch < total_epoch else epoch-total_epoch, 0.0)
    need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch-epoch), True))
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

    network.set_drop_path(float(epoch+1) / total_epoch, xargs.drop_path_rate)
    if xargs.algo == 'gdas':
      network.set_tau( xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (total_epoch-1) )
      logger.log('[RESET tau as : {:} and drop_path as {:}]'.format(network.tau, network.drop_path))
    if epoch < total_epoch: # Use all archs as usual in SPOS
      archs_to_sample_from = None
    elif epoch >= total_epoch and xargs.greedynas_epochs > 0:
      if epoch == total_epoch:
        logger.log(f"About to start GreedyNAS supernet training with archs(len={len(greedynas_archs)}), head={[api.archstr2index[x.tostr()] for x in greedynas_archs[0:10]]}")
      archs_to_sample_from = greedynas_archs
    search_w_loss, search_w_top1, search_w_top5, search_a_loss, search_a_top1, search_a_top5, supernet_metrics, arch_overview \
                = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, xargs.algo, logger, 
                  smoke_test=xargs.dry_run, meta_learning=xargs.meta_learning, api=api, epoch=epoch,
                  supernets_decomposition=supernets_decomposition, arch_groups_quartiles=arch_groups_quartiles, arch_groups_brackets=arch_groups_brackets,
                  all_archs=archs_to_sample_from, grad_metrics_percentiles=grad_metrics_percs, 
                  percentiles=percentiles, metrics_percs=metrics_percs, args=xargs, replay_buffer=replay_buffer)
    if xargs.replay_buffer is not None and xargs.replay_buffer > 0:
      # Use the lowest-loss architectures from last epoch as replay buffer for the subsequent epoch
      arch_metrics = sorted(zip(arch_overview["all_archs"], arch_overview[xargs.replay_buffer_metric]), key = lambda x: x[1])
      replay_buffer = [x[0] for x in arch_metrics[-int(args.replay_buffer):]]

    for percentile in arch_perf_percs.keys(): # Finds a threshold for each performance bracket from the latest epoch so that we can do exploiting search later
      arch_perf_percs[percentile] = arch_overview["train_loss"][min(math.floor(len(arch_overview["train_loss"]) * (percentile/100)), len(arch_overview["train_loss"])-1)]
    grad_log_keys = ["gn", "gnL1", "sogn", "sognL1", "grad_normalized", "grad_accum", "grad_accum_singleE", "grad_accum_decay", "grad_mean_accum", "grad_mean_sign", "grad_var_accum", "grad_var_decay_accum"]
    if xargs.supernets_decomposition:
      for percentile in percentiles[1:]:
        for log_key in grad_log_keys:
          metrics_percs[supernet_key+"_"+log_key]["perc"+str(percentile)][epoch].append(grad_metrics_percs["perc"+str(percentile)]["supernet"][log_key])
      
    search_time.update(time.time() - start_time)
    logger.log('[{:}] search [base] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
    logger.log('[{:}] search [arch] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, search_a_loss, search_a_top1, search_a_top5))
    if xargs.algo == 'enas':
      ctl_loss, ctl_acc, baseline, ctl_reward \
                                 = train_controller(valid_loader, network, criterion, a_optimizer, baseline, epoch_str, xargs.print_freq, logger)
      logger.log('[{:}] controller : loss={:}, acc={:}, baseline={:}, reward={:}'.format(epoch_str, ctl_loss, ctl_acc, baseline, ctl_reward))

    genotype, temp_accuracy = get_best_arch(train_loader, valid_loader, network, xargs.eval_candidate_num, xargs.algo, xargs=xargs, criterion=criterion, logger=logger, api=api)
    if xargs.algo == 'setn' or xargs.algo == 'enas':
      network.set_cal_mode('dynamic', genotype)
    elif xargs.algo == 'gdas':
      network.set_cal_mode('gdas', None)
    elif xargs.algo.startswith('darts'):
      network.set_cal_mode('joint', None)
    elif 'random' in xargs.algo:
      network.set_cal_mode('urs', None)
    else:
      raise ValueError('Invalid algorithm name : {:}'.format(xargs.algo))
    logger.log('[{:}] - [get_best_arch] : {:} -> {:}'.format(epoch_str, genotype, temp_accuracy))
    valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion, xargs.algo, logger, steps=500 if xargs.dataset=="cifar5m" else None)
    logger.log('[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}% | {:}'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5, genotype))
    valid_accuracies[epoch] = valid_a_top1

    if hasattr(search_loader.sampler, "reset_counter"):
      search_loader.sampler.counter += 1

    genotypes[epoch] = genotype
    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    save_path = save_checkpoint({'epoch' : epoch + 1,
                'args'  : deepcopy(xargs),
                'baseline'    : baseline,
                'search_model': search_model.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'a_optimizer' : a_optimizer.state_dict(),
                'w_scheduler' : w_scheduler.state_dict(),
                'genotypes'   : genotypes,
                'valid_accuracies' : valid_accuracies,
                "grad_metrics_percs" : grad_metrics_percs,
                "archs_subset" : archs_subset},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
        }, logger.path('info'), logger)
    with torch.no_grad():
      logger.log('{:}'.format(search_model.show_alphas()))
    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch], '200')))
    for batch_idx in range(len(search_loader)):
      interim = {}
      for metric in supernet_metrics.keys():
        for bracket in supernet_metrics[metric].keys():
          interim[metric+"."+bracket] = supernet_metrics[metric][bracket][batch_idx]
      wandb.log({**interim, "epoch":epoch, "batch":batch_idx, "true_step":epoch*len(search_loader)+batch_idx})


    to_log = {"loss_w":search_w_loss, "loss_a":search_a_loss, "acc_w":search_w_top1, "acc_a":search_a_top1, "epoch":epoch, 
      "final": summarize_results_by_dataset(genotype, api=api, iepoch=199, hp='200')}
    if xargs.supernets_decomposition:
      # interim = {"perc"+str(percentile):{} for percentile in percentiles}
      interim = {supernet_key+"_" + key:{} for key in grad_log_keys}
      for percentile in percentiles[1:]:
        for key in grad_log_keys:
          interim[supernet_key+"_"+key]["perc"+str(percentile)] = metrics_percs[supernet_key+"_"+key]["perc"+str(percentile)][epoch][-1] # NOTE the last list should have only one item regardless
      to_log = {**to_log, **interim}

      grad_metrics_percs["grad_accum_singleE"] = None
      grad_metrics_percs["grad_accum_singleE_tensor"] = None

    wandb.log(to_log)


    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  wandb.log({"supernet_train_time":search_time.sum})

  # the final post procedure : count the time
  start_time = time.time()
  gpu_mem = torch.cuda.get_device_properties(0).total_memory
  train_data_postnet, valid_data_postnet, xshape_postnet, class_num_postnet = get_datasets(xargs.dataset_postnet, xargs.data_path, -1, mmap=xargs.mmap, total_samples=xargs.total_samples)

  search_loader_postnet, train_loader_postnet, valid_loader_postnet = get_nas_search_loaders(train_data_postnet, valid_data_postnet, xargs.dataset_postnet, 'configs/nas-benchmark/', 
    (resolved_train_batch_size, resolved_val_batch_size), workers=xargs.workers, valid_ratio=xargs.val_dset_ratio, determinism=xargs.deterministic_loader, 
    meta_learning=xargs.meta_learning, epochs=xargs.eval_epochs, merge_train_val=xargs.merge_train_val_postnet, merge_train_val_and_use_test = xargs.merge_train_val_and_use_test)
  _, train_loader_stats, val_loader_stats = get_nas_search_loaders(train_data_postnet, valid_data_postnet, xargs.dataset_postnet, 'configs/nas-benchmark/', 
    (128 if gpu_mem < 8147483648 else 1024, 128 if gpu_mem < 8147483648 else 1024), workers=xargs.workers, valid_ratio=xargs.val_dset_ratio, determinism="all", 
    meta_learning=xargs.meta_learning, epochs=xargs.eval_epochs, merge_train_val=xargs.merge_train_val_postnet, merge_train_val_and_use_test = xargs. merge_train_val_and_use_test)

  if xargs.cand_eval_method in ['val_acc', 'val']:
    genotype, temp_accuracy = get_best_arch(train_loader_postnet, valid_loader_postnet, network, xargs.eval_candidate_num, xargs.algo, criterion=criterion, logger=logger, style=xargs.cand_eval_method, api=api)
  elif xargs.cand_eval_method == 'sotl': #TODO probably get rid of this
    if greedynas_archs is None: # TODO might want to implement some greedy sampling here? None will just sample uniformly as in SPOS
      logger.log("Since greedynas_archs=None, we will sample archs anew for get_best_arch")
      archs_to_sample_from = None
    else:
      archs_to_sample_from = greedynas_archs
      logger.log(f"Reusing greedynas_archs for get_best_arch with head = {[api.archstr2index[x.tostr()] for x in archs_to_sample_from]}")

    genotype, temp_accuracy = get_best_arch(train_loader_postnet, valid_loader_postnet, network, xargs.eval_candidate_num, xargs.algo, criterion=criterion, logger=logger, style=xargs.cand_eval_method, 
      w_optimizer=w_optimizer, w_scheduler=w_scheduler, config=config, epochs=xargs.eval_epochs, steps_per_epoch=xargs.steps_per_epoch, 
      api=api, additional_training = xargs.additional_training, val_loss_freq=xargs.val_loss_freq, 
      overwrite_additional_training=xargs.overwrite_additional_training, scheduler_type=xargs.scheduler, xargs=xargs, train_loader_stats=train_loader_stats, val_loader_stats=val_loader_stats, 
      model_config=model_config, all_archs=archs_to_sample_from)

  if xargs.algo == 'setn' or xargs.algo == 'enas':
    network.set_cal_mode('dynamic', genotype)
  elif xargs.algo == 'gdas':
    network.set_cal_mode('gdas', None)
  elif xargs.algo.startswith('darts'):
    network.set_cal_mode('joint', None)
  elif 'random' in xargs.algo:
    network.set_cal_mode('urs', None)
  else:
    raise ValueError('Invalid algorithm name : {:}'.format(xargs.algo))
  search_time.update(time.time() - start_time)

  valid_a_loss , valid_a_top1 , valid_a_top5 = valid_func(valid_loader_postnet, network, criterion, xargs.algo, logger)
  logger.log('Last : the gentotype is : {:}, with the validation accuracy of {:.3f}%.'.format(genotype, valid_a_top1))

  logger.log('\n' + '-'*100)
  # check the performance from the architecture dataset
  logger.log('[{:}] run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(xargs.algo, total_epoch, search_time.sum, genotype))
  if api is not None: logger.log('{:}'.format(api.query_by_arch(genotype, '200') ))
  results_by_dataset = summarize_results_by_dataset(genotype, api, separate_mean_std=False)
  wandb.log(results_by_dataset)
  logger.close()
  



if __name__ == '__main__':
  parser = argparse.ArgumentParser("Weight sharing NAS methods to search for cells.")
  parser.add_argument('--data_path'   ,       type=str,   help='Path to dataset')
  parser.add_argument('--dataset'     ,       type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120', 'cifar5m'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--search_space',       type=str,   default='tss', choices=['tss'], help='The search space name.')
  parser.add_argument('--algo'        ,       type=str,   help='The search space name.')
  parser.add_argument('--use_api'     ,       type=int,   default=1, choices=[0,1], help='Whether use API or not (which will cost much memory).')
  # FOR GDAS
  parser.add_argument('--tau_min',            type=float, default=0.1,  help='The minimum tau for Gumbel Softmax.')
  parser.add_argument('--tau_max',            type=float, default=10,   help='The maximum tau for Gumbel Softmax.')
  # channels and number-of-cells
  parser.add_argument('--max_nodes'   ,       type=int,   default=4,  help='The maximum number of nodes.')
  parser.add_argument('--channel'     ,       type=int,   default=16, help='The number of channels.')
  parser.add_argument('--num_cells'   ,       type=int,   default=5,  help='The number of cells in one stage.')
  #
  parser.add_argument('--eval_candidate_num', type=int,   default=100, help='The number of selected architectures to evaluate.')
  #
  parser.add_argument('--track_running_stats',type=int,   default=0, choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--affine'      ,       type=int,   default=0, choices=[0,1],help='Whether use affine=True or False in the BN layer.')
  parser.add_argument('--config_path' ,       type=str,   default='./configs/nas-benchmark/algos/weight-sharing.config', help='The path of configuration.')
  parser.add_argument('--overwite_epochs',    type=int,   help='The number of epochs to overwrite that value in config files.')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay' , type=float, default=1e-3, help='weight decay for arch encoding')
  parser.add_argument('--arch_eps'          , type=float, default=1e-8, help='weight decay for arch encoding')
  parser.add_argument('--drop_path_rate'  ,  type=float, help='The drop path rate.')
  # log
  parser.add_argument('--workers',            type=int,   default=0,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   default='./output/search', help='Folder to save checkpoints and log.')
  parser.add_argument('--print_freq',         type=int,   default=200,  help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  parser.add_argument('--cand_eval_method',          type=str,   help='SoTL or ValAcc', default='val_acc', choices = ['sotl', 'val_acc', 'val'])
  parser.add_argument('--sotl_dataset_eval',          type=str,   help='Whether to do the SoTL short training on the train+val dataset or the test set', default='train', choices = ['train_val', "train", 'test'])
  parser.add_argument('--sotl_dataset_train',          type=str,   help='TODO doesnt work currently. Whether to do the train step in SoTL on the whole train dataset (ie. the default split of CIFAR10 to train/test) or whether to use the extra split of train into train/val', 
    default='train', choices = ['train_val', 'train'])
  parser.add_argument('--steps_per_epoch',           default=100,  help='Number of minibatches to train for when evaluating candidate architectures with SoTL')
  parser.add_argument('--eval_epochs',          type=int, default=1,   help='Number of epochs to train for when evaluating candidate architectures with SoTL')
  parser.add_argument('--additional_training',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=True,   help='Whether to train the supernet samples or just go through the training loop with no grads')
  parser.add_argument('--val_batch_size',          type=int, default=64,   help='Batch size for the val loader - this is crucial for SoVL and similar experiments, but bears no importance in the standard NASBench setup')
  parser.add_argument('--dry_run',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False,   help='WANDB dry run - whether to sync to the cloud')
  parser.add_argument('--val_dset_ratio',          type=float, default=1,   help='Only uses a ratio of X for the valid data loader. Used for testing SoValAcc robustness')
  parser.add_argument('--val_loss_freq',          type=int, default=1,   help='How often to calculate val loss during training. Probably better to only this for smoke tests as it is generally better to record all and then post-process if different results are desired')
  parser.add_argument('--overwrite_additional_training',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False,   help='Whether to load checkpoints of additional training')
  parser.add_argument('--scheduler',          type=str, default=None,   help='Whether to use different training protocol for the postnet training')
  parser.add_argument('--train_batch_size',          type=int, default=None,   help='Training batch size for the POST-SUPERNET TRAINING!')
  parser.add_argument('--lr',          type=float, default=None,   help='Constant LR for the POST-SUPERNET TRAINING!')
  parser.add_argument('--deterministic_loader',          type=str, default='all', choices=['None', 'train', 'val', 'all'],   help='Whether to choose SequentialSampler or RandomSampler for data loaders')
  parser.add_argument('--reinitialize',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to use trained supernetwork weights for initialization')
  parser.add_argument('--meta_learning',          type=str, default="", help='Whether to split training data per classes (ie. classes 0-5 into train, 5-10 into validation set and/or use val set for training arch')
  parser.add_argument('--individual_logs',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=True, help='Whether to log each of the eval_candidate_num sampled architectures as a separate WANDB run')
  parser.add_argument('--total_estimator_steps',          type=int, default=10, help='Number of batches for evaluating the total_val/total_train etc. metrics')
  parser.add_argument('--corrs_freq',          type=int, default=4, help='Calculate corrs based on every i-th minibatch')
  parser.add_argument('--mmap',          type=str, default=None, help='Whether to mmap cifar5m')
  parser.add_argument('--search_epochs',          type=int, default=None, help='Can be used to explicitly set the number of search epochs')
  parser.add_argument('--size_percentile',          type=float, default=None, help='Percentile of arch param count in NASBench sampling, ie. 0.9 will give top 10% archs by param count only')
  parser.add_argument('--total_samples',          type=int, default=None, help='Number of total samples in dataset. Useful for limiting Cifar5m')
  parser.add_argument('--restart',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=None, help='WHether to force or disable restart of training via must_restart')
  parser.add_argument('--grads_analysis',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='WHether to force or disable restart of training via must_restart')
  parser.add_argument('--perf_percentile',          type=float, default=None, help='Perf percentile of architectures to sample from')
  parser.add_argument('--resample',          type=str, default=False, help='Only makes sense when also using reinitialize')
  parser.add_argument('--supernets_decomposition',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Track updates to supernetwork by quartile')
  parser.add_argument('--supernets_decomposition_mode',          type=str, choices=["perf", "size"], default="perf", help='Track updates to supernetwork by quartile')
  parser.add_argument('--supernets_decomposition_topk',          type=int, default=-1, help='How many archs to sample from the search space')
  parser.add_argument('--evenify_training',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Since subnetworks might come out unevenly trained, we can set a standard number of epochs-equivalent-of-trianing-from-scratch and match that for each')
  parser.add_argument('--adaptive_lr',          type=lambda x: False if x in ["False", "false", "", "None"] else x, choices=["custom", "1cycle"], default=False, help='Do a quick search for best LR before post-supernet training')
  parser.add_argument('--sandwich',          type=int, default=None, help='Do a quick search for best LR before post-supernet training')
  parser.add_argument('--sandwich_mode',          type=str, default=None, help='Do a quick search for best LR before post-supernet training')
  parser.add_argument('--force_rewrite',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Load saved seed or not')
  parser.add_argument('--greedynas_epochs',          type=int, default=None, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--greedynas_lr',          type=float, default=0.01, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--merge_train_val_postnet',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--merge_train_val_supernet',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--postnet_switch_train_val',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--dataset_postnet',          type=str, default=None, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'cifar5m'], help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--reptile',          type=int, default=None, help='How many steps to do in Reptile rollout')
  parser.add_argument('--reptile_weight',          type=float, default=1., help='Interpolation coefficient for Reptile')
  parser.add_argument('--replay_buffer',          type=int, default=None, help='Replay buffer to tackle multi-model forgetting')
  parser.add_argument('--replay_buffer_mode',          type=str, default="random", choices=["random", "perf", "size", None], help='How to figure out what to put in the replay buffer')
  parser.add_argument('--replay_buffer_percentile',          type=float, default=0.9, help='Replay buffer percentile of performance etc.')
  parser.add_argument('--replay_buffer_weight',          type=float, default=0.5, help='Trade off between new arch loss and buffer loss')
  parser.add_argument('--replay_buffer_metric',          type=str, default="train_loss", choices=["train_loss", "train_acc", "val_acc", "val_loss"], help='Trade off between new arch loss and buffer loss')
  parser.add_argument('--evenly_split',          type=str, default=None, choices=["perf", "size"], help='Whether to split the NASBench archs into eval_candidate_num brackets and then take an arch from each bracket to ensure they are not too similar')
  parser.add_argument('--merge_train_val_and_use_test',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Merges CIFAR10 train/val into one (ie. not split in half) AND then also treats test set as validation')


  args = parser.parse_args()

  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'
  mp.set_start_method('spawn')
  wandb_auth()
  run = wandb.init(project="NAS", group=f"Search_Cell_{args.algo}", reinit=True)

  if 'TORCH_HOME' not in os.environ:
    if os.path.exists('/notebooks/storage/.torch/'):
      os.environ["TORCH_HOME"] = '/notebooks/storage/.torch/'

    gdrive_torch_home = "/content/drive/MyDrive/colab/data/TORCH_HOME"

    if os.path.exists(gdrive_torch_home):
      os.environ["TORCH_HOME"] = "/content/drive/MyDrive/colab/data/TORCH_HOME"
  
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  if args.overwite_epochs is None:
    args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space),
        args.dataset,
        '{:}-affine{:}_BN{:}-{:}'.format(args.algo, args.affine, args.track_running_stats, args.drop_path_rate))
  else:
    args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space),
        args.dataset,
        '{:}-affine{:}_BN{:}-E{:}-{:}'.format(args.algo, args.affine, args.track_running_stats, args.overwite_epochs, args.drop_path_rate))


  wandb.config.update(args)

  main(args)
