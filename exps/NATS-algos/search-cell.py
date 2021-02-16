##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v1 --rand_seed 777
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
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch 5 --train_batch_size 128 --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 32 --scheduler cos_adjusted --overwrite_additional_training True --dry_run True
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch None --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 64 --dry_run=True --train_batch_size 512
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 3 --cand_eval_method sotl --steps_per_epoch None --eval_epochs 1
# python ./exps/NATS-algos/search-cell.py --algo=random --cand_eval_method=sotl --data_path=$TORCH_HOME/cifar.python --dataset=cifar10 --eval_epochs=2 --rand_seed=2 --steps_per_epoch=None
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo random
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo random
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
######################################################################################
import os, sys, time, random, argparse
import numpy as np
from copy import deepcopy
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
  calculate_valid_acc_single_arch, calculate_valid_accs, 
  calc_corrs_after_dfs, calc_corrs_val, get_true_rankings, SumOfWhatever)
import wandb
import itertools
import scipy.stats
import time
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


def backward_step_unrolled(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets):
  # _compute_unrolled_model
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


def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, algo, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    base_inputs = base_inputs.cuda(non_blocking=True)
    arch_inputs = arch_inputs.cuda(non_blocking=True)
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # Update the weights
    if algo == 'setn':
      sampled_arch = network.dync_genotype(True)
      network.set_cal_mode('dynamic', sampled_arch)
    elif algo == 'gdas':
      network.set_cal_mode('gdas', None)
    elif algo.startswith('darts'):
      network.set_cal_mode('joint', None)
    elif algo == 'random':
      network.set_cal_mode('urs', None)
    elif algo == 'enas':
      with torch.no_grad():
        network.controller.eval()
        _, _, sampled_arch = network.controller()
      network.set_cal_mode('dynamic', sampled_arch)
    else:
      raise ValueError('Invalid algo name : {:}'.format(algo))
      
    network.zero_grad()
    _, logits = network(base_inputs)
    base_loss = criterion(logits, base_targets)
    base_loss.backward()
    w_optimizer.step()
    # record

    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))

    # update the architecture-weight
    if algo == 'setn':
      network.set_cal_mode('joint')
    elif algo == 'gdas':
      network.set_cal_mode('gdas', None)
    elif algo.startswith('darts'):
      network.set_cal_mode('joint', None)
    elif algo == 'random':
      network.set_cal_mode('urs', None)
    elif algo != 'enas':
      raise ValueError('Invalid algo name : {:}'.format(algo))
    network.zero_grad()
    if algo == 'darts-v2':
      arch_loss, logits = backward_step_unrolled(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets)
      a_optimizer.step()
    elif algo == 'random' or algo == 'enas':
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

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


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

def get_best_arch(train_loader, valid_loader, network, n_samples, algo, logger, 
  additional_training=True, api=None, style:str='val_acc', w_optimizer=None, w_scheduler=None, 
  config: Dict=None, epochs:int=1, steps_per_epoch:int=100, 
  val_loss_freq:int=1, overwrite_additional_training:bool=False, 
  scheduler_type:str=None, xargs:Namespace=None):
  with torch.no_grad():
    network.eval()
    if algo == 'random':
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

    # The true rankings are used to calculate correlations later

    true_rankings, final_accs = get_true_rankings(archs, api)
    
    corr_funs = {"kendall": lambda x,y: scipy.stats.kendalltau(x,y).correlation, 
      "spearman":lambda x,y: scipy.stats.spearmanr(x,y).correlation, 
      "pearson":lambda x, y: scipy.stats.pearsonr(x,y)[0]}
    if steps_per_epoch is not None and steps_per_epoch != "None":
      steps_per_epoch = int(steps_per_epoch)
    elif steps_per_epoch in [None, "None"]:
      steps_per_epoch = len(train_loader)
    else:
      raise NotImplementedError

    if style == 'val_acc':
      decision_metrics = calculate_valid_accs(xloader=valid_loader, archs=archs, network=network)
      corr_per_dataset = calc_corrs_val(archs=archs, valid_accs=decision_metrics, final_accs=final_accs, true_rankings=true_rankings, corr_funs=corr_funs)

      wandb.log(corr_per_dataset)

  if style == 'sotl' or style == "sovl":    
    # Simulate short training rollout to compute SoTL for candidate architectures
    cond = logger.path('corr_metrics').exists() and not overwrite_additional_training
    metrics_keys = ["sotl", "val", "sovl", "sovalacc", "sotrainacc", "sovalacc_top5", "sotrainacc_top5", "train_losses", "val_losses", "total_val"]
    must_restart = False
    start_arch_idx = 0

    if cond:
      logger.log("=> loading checkpoint of the last-checkpoint '{:}' start".format(logger.path('corr_metrics')))

      checkpoint = torch.load(logger.path('corr_metrics'))
      checkpoint_config = checkpoint["config"] if "config" in checkpoint.keys() else {}

      try:
        if type(list(checkpoint["metrics"]["sotl"].keys())[0]) is not str:
          must_restart = True # will need to restart metrics because using the old checkpoint format
        metrics = {k:checkpoint["metrics"][k] if k in checkpoint["metrics"] else {} for k in metrics_keys}

        prototype = metrics[metrics_keys[0]]
        first_arch = next(iter(metrics[metrics_keys[0]].keys()))
        for metric_key in metrics_keys:
          if not (len(metrics[metric_key]) == len(prototype) and len(metrics[metric_key][first_arch]) == len(prototype[first_arch])):
            must_restart = True
      except:
        must_restart = True


      
      decision_metrics = checkpoint["decision_metrics"] if "decision_metrics" in checkpoint.keys() else []
      start_arch_idx = checkpoint["start_arch_idx"]
      cond1={k:v for k,v in checkpoint_config.items() if ('path' not in k and 'dir' not in k and k not in ["dry_run"])}
      cond2={k:v for k,v in vars(xargs).items() if ('path' not in k and 'dir' not in k and k not in ["dry_run"])}
      logger.log(f"Checkpoint config: {cond1}")
      logger.log(f"Newly input config: {cond2}")
      if (cond1 == cond2):
        logger.log("Both configs are equal.")
      else:
        logger.log("Checkpoint and current config are not the same! need to restart")
        different_items = {k: cond1[k] for k in cond1 if k in cond2 and cond1[k] != cond2[k]}
        logger.log(f"Different items are : {different_items}")


    if (not cond) or must_restart or (xargs is None) or (cond1 != cond2) or any([len(x) == 0 for x in metrics.values()]): #config should be an ArgParse Namespace
      if not cond:
        logger.log(f"Did not find a checkpoint for supernet post-training at {logger.path('corr_metrics')}")

      else:
        logger.log(f"Starting postnet training with fresh metrics")
    
      metrics = {k:{arch.tostr():[[] for _ in range(epochs)] for arch in archs} for k in metrics_keys}       
      start_arch_idx = 0


    train_start_time = time.time()

    train_stats = [[] for _ in range(epochs*steps_per_epoch+1)]

    for arch_idx, sampled_arch in tqdm(enumerate(archs[start_arch_idx:], start_arch_idx), desc="Iterating over sampled architectures", total = n_samples-start_arch_idx):
      network2 = deepcopy(network)
      network2.set_cal_mode('dynamic', sampled_arch)
      if scheduler_type in ['linear_warmup', 'linear']:
        config = config._replace(scheduler=scheduler_type, warmup=1, LR_min=0)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type == "cos_reinit":
        # In practice, this leads to constant LR = 0.025 since the original Cosine LR is annealed over 100 epochs and our training schedule is very short
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type in ['cos_adjusted']:
        config = config._replace(scheduler='cos', warmup=0, epochs=epochs)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      else:
        # NOTE in practice, since the Search function uses Cosine LR with T_max that finishes at end of search_func training, this switches to a constant 1e-3 LR.
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
        w_optimizer2.load_state_dict(w_optimizer.state_dict())
        w_scheduler2.load_state_dict(w_scheduler.state_dict())

      if arch_idx == start_arch_idx: #Should only print it once at the start of training
        logger.log(f"Optimizers for the supernet post-training: {w_optimizer2}, {w_scheduler2}")

      running_sotl = 0 # TODO implement better SOTL class to make it more adjustible and get rid of this repeated garbage everywhere
      running_sovl = 0
      running_sovalacc = 0
      running_sotrainacc = 0
      running_sovalacc_top5 = 0
      running_sotrainacc_top5 = 0

      _, val_acc_total, _ = valid_func(xloader=valid_loader, network=network2, criterion=criterion, algo=algo, logger=logger)

      true_step = 0
      arch_str = sampled_arch.tostr()

      if steps_per_epoch is None or steps_per_epoch=="None":
        steps_per_epoch = len(train_loader)

      q = mp.Queue()
      # This reporting process is necessary due to WANDB technical difficulties. It is used to continuously report train stats from a separate process
      # Otherwise, when a Run is intiated from a Sweep, it is not necessary to log the results to separate training runs. But that it is what we want for the individual arch stats
      p=mp.Process(target=train_stats_reporter, kwargs=dict(queue=q, config=vars(xargs),
          sweep_group=f"Search_Cell_{algo}_arch", sweep_run_name=wandb.run.name or wandb.run.id or "unknown", arch=sampled_arch.tostr()))
      p.start()

      for epoch_idx in range(epochs):

        if epoch_idx == 0:
          metrics["total_val"][arch_str][epoch_idx] = [val_acc_total]*(len(train_loader)-1)
        else:
          metrics["total_val"][arch_str][epoch_idx] = [metrics["total_val"][arch_str][epoch_idx-1][-1]]*(len(train_loader)-1)

        valid_loader_iter = iter(valid_loader) if not additional_training else None # This causes deterministic behavior for validation data since the iterator gets passed in to each function

        for batch_idx, data in enumerate(train_loader):

          if (steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx > steps_per_epoch:
            break
          with torch.set_grad_enabled(mode=additional_training):
            if scheduler_type in ["linear", "linear_warmup"]:
              w_scheduler2.update(epoch_idx, 1.0 * batch_idx / min(len(train_loader), steps_per_epoch))

            elif scheduler_type == "cos_adjusted":
              w_scheduler2.update(epoch_idx , batch_idx/min(len(train_loader), steps_per_epoch))
            else:
              w_scheduler2.update(epoch_idx, 0.0)


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
            
          true_step += 1

          if batch_idx == 0 or (batch_idx % val_loss_freq == 0):
            valid_acc, valid_acc_top5, valid_loss = calculate_valid_acc_single_arch(valid_loader=valid_loader, arch=sampled_arch, network=network2, criterion=criterion, valid_loader_iter=valid_loader_iter)
          
          batch_train_stats = {"lr":w_scheduler2.get_lr()[0], "true_step":true_step, "train_loss":loss.item(), "train_acc_top1":train_acc_top1.item(), "train_acc_top5":train_acc_top5.item(), 
            "valid_loss":valid_loss, "valid_acc":valid_acc, "valid_acc_top5":valid_acc_top5}
          # print(epoch_idx*steps_per_epoch+batch_idx)
          q.put(batch_train_stats)
          train_stats[epoch_idx*steps_per_epoch+batch_idx].append(batch_train_stats)

          # wandb.log(batch_train_stats)
          running_sovl -= valid_loss
          running_sovalacc += valid_acc
          running_sovalacc_top5 += valid_acc_top5
          running_sotl -= loss.item() # Need to have negative loss so that the ordering is consistent with val acc
          running_sotrainacc += train_acc_top1.item()
          running_sotrainacc_top5 += train_acc_top5.item()

          metrics["sotl"][arch_str][epoch_idx].append(running_sotl)
          metrics["val"][arch_str][epoch_idx].append(valid_acc)
          metrics["sovl"][arch_str][epoch_idx].append(running_sovl)
          metrics["sovalacc"][arch_str][epoch_idx].append(running_sovalacc)
          metrics["sotrainacc"][arch_str][epoch_idx].append(running_sotrainacc)
          metrics["sovalacc_top5"][arch_str][epoch_idx].append(running_sovalacc_top5)
          metrics["sotrainacc_top5"][arch_str][epoch_idx].append(running_sotrainacc_top5)
          metrics["train_losses"][arch_str][epoch_idx].append(-loss.item())
          metrics["val_losses"][arch_str][epoch_idx].append(-valid_loss)
        
        if additional_training:
          _, val_acc_total, _ = valid_func(xloader=valid_loader, network=network2, criterion=criterion, algo=algo, logger=logger)

        metrics["total_val"][arch_str][epoch_idx].append(val_acc_total)

      final_metric = None # Those final/decision metrics are not very useful apart from being a compatibility layer with how get_best_arch worked in the base repo
      if style == "sotl":
        final_metric = running_sotl
      elif style == "sovl":
        final_metric = running_sovl
      decision_metrics.append(final_metric)

      corr_metrics_path = save_checkpoint({"corrs":{}, "metrics":metrics, 
        "archs":archs, "start_arch_idx": arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},   
        logger.path('corr_metrics'), logger, quiet=True)

      q.put("SENTINEL") # This lets the Reporter process know it should quit
      # p.join()

            
    train_total_time = time.time()-train_start_time
    print(f"Train total time: {train_total_time}")

    wandb.run.summary["train_total_time"] = train_total_time

    metrics_FD = {k+"FD": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=1).get_time_series(chunked=True, mode="fd") for arch in archs} for k,v in metrics.items() if k in ['val', 'train_losses', 'val_losses']}
    metrics.update(metrics_FD)
    
    if epochs > 1:
      metrics_E1 = {k+"E1": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=1).get_time_series(chunked=True) for arch in archs} for k,v in metrics.items()}
      metrics.update(metrics_E1)
    
    
    # print(metrics["val_losses"])
    # print(metrics["val_lossesE1"])
    # print(metrics["sovl"])

    start=time.time()
    corrs = {}
    to_logs = []
    for k,v in tqdm(metrics.items(), desc="Calculating correlations"):
      # We cannot do logging synchronously with training becuase we need to know the results of all archs for i-th epoch before we can log correlations for that epoch
      corr, to_log = calc_corrs_after_dfs(epochs=epochs, xloader=train_loader, steps_per_epoch=steps_per_epoch, metrics_depth_dim=v, 
    final_accs = final_accs, archs=archs, true_rankings = true_rankings, corr_funs=corr_funs, prefix=k, api=api, wandb_log=False)
      corrs["corrs_"+k] = corr
      to_logs.append(to_log)

    print(f"Calc corrs time: {time.time()-start}")

    if n_samples-start_arch_idx > 0: #If there was training happening - might not be the case if we just loaded checkpoint
      # We reshape the stored train statistics so that it is a Seq[Dict[k: summary statistics across all archs for a timestep]] instead of Seq[Seq[Dict[k: train stat for a single arch]]]
      processed_train_stats = []
      stats_keys = batch_train_stats.keys()
      for idx, stats_across_time in tqdm(enumerate(train_stats), desc="Processing train stats"):
        agg = {k: np.array([single_train_stats[k] for single_train_stats in stats_across_time]) for k in stats_keys}
        agg = {k: {"mean":np.mean(v), "std": np.std(v)} for k,v in agg.items()}
        agg["true_step"] = idx
        processed_train_stats.append(agg)


    for epoch_idx in range(len(to_logs[0])):
      relevant_epochs = [to_logs[i][epoch_idx] for i in range(len(to_logs))]
      for batch_idx in range(len(relevant_epochs[0])):
        relevant_batches = [relevant_epoch[batch_idx] for relevant_epoch in relevant_epochs]
        all_batch_data = {}
        for batch in relevant_batches:
          all_batch_data.update(batch)

        # Here we log both the aggregated train statistics and the correlations
        if n_samples-start_arch_idx > 0: #If there was training happening - might not be the case if we just loaded checkpoint
          all_data_to_log = {**all_batch_data, **processed_train_stats[epoch_idx*steps_per_epoch+batch_idx]}
        else:
          all_data_to_log = all_batch_data
        wandb.log(all_data_to_log)
  
  if style in ["sotl", "sovl"] and n_samples-start_arch_idx > 0: # otherwise, we are just reloading the previous checkpoint so should not save again
    corr_metrics_path = save_checkpoint({"metrics":metrics, "corrs": corrs, 
      "archs":archs, "start_arch_idx":arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},
      logger.path('corr_metrics'), logger)
    try:
      wandb.save(str(corr_metrics_path.absolute()))
    except:
      print("Upload to WANDB failed")

  best_idx = np.argmax(decision_metrics)
  best_arch, best_valid_acc = archs[best_idx], decision_metrics[best_idx]
  return best_arch, best_valid_acc


def valid_func(xloader, network, criterion, algo, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  with torch.no_grad():
    network.eval()
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      arch_targets = arch_targets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - end)
      # prediction
      _, logits = network(arch_inputs.cuda(non_blocking=True))
      arch_loss = criterion(logits, arch_targets)
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
  network.train()
  return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  if xargs.overwite_epochs is None:
    extra_info = {'class_num': class_num, 'xshape': xshape}
  else:
    extra_info = {'class_num': class_num, 'xshape': xshape, 'epochs': xargs.overwite_epochs}
  config = load_config(xargs.config_path, extra_info, logger)
  resolved_train_batch_size, resolved_val_batch_size = xargs.train_batch_size if xargs.train_batch_size is not None else config.batch_size, xargs.val_batch_size if xargs.val_batch_size is not None else config.test_batch_size
  search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', 
    (resolved_train_batch_size, resolved_val_batch_size), 0, valid_ratio=xargs.val_dset_ratio)
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
  # TODO this logging search omdel makes a big mess in the logs! Although it is thecnically useful information
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

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  network, criterion = search_model.cuda(), criterion.cuda()  # use a single GPU

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    if os.name == 'nt': # The last-info pickles have PosixPaths serialized in them, hence they cannot be instantied on Windows
      import pathlib
      temp = pathlib.PosixPath
      pathlib.PosixPath = pathlib.WindowsPath
    last_info   = torch.load(last_info.resolve())
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
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: network.return_topK(1, True)[0]}
    baseline = None

  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  for epoch in range(start_epoch, total_epoch):
    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch-epoch), True))
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

    network.set_drop_path(float(epoch+1) / total_epoch, xargs.drop_path_rate)
    if xargs.algo == 'gdas':
      network.set_tau( xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (total_epoch-1) )
      logger.log('[RESET tau as : {:} and drop_path as {:}]'.format(network.tau, network.drop_path))
    search_w_loss, search_w_top1, search_w_top5, search_a_loss, search_a_top1, search_a_top5 \
                = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, xargs.algo, logger)
    search_time.update(time.time() - start_time)
    logger.log('[{:}] search [base] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
    logger.log('[{:}] search [arch] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, search_a_loss, search_a_top1, search_a_top5))
    if xargs.algo == 'enas':
      ctl_loss, ctl_acc, baseline, ctl_reward \
                                 = train_controller(valid_loader, network, criterion, a_optimizer, baseline, epoch_str, xargs.print_freq, logger)
      logger.log('[{:}] controller : loss={:}, acc={:}, baseline={:}, reward={:}'.format(epoch_str, ctl_loss, ctl_acc, baseline, ctl_reward))

    genotype, temp_accuracy = get_best_arch(train_loader, valid_loader, network, xargs.eval_candidate_num, xargs.algo, logger=logger, api=api)
    if xargs.algo == 'setn' or xargs.algo == 'enas':
      network.set_cal_mode('dynamic', genotype)
    elif xargs.algo == 'gdas':
      network.set_cal_mode('gdas', None)
    elif xargs.algo.startswith('darts'):
      network.set_cal_mode('joint', None)
    elif xargs.algo == 'random':
      network.set_cal_mode('urs', None)
    else:
      raise ValueError('Invalid algorithm name : {:}'.format(xargs.algo))
    logger.log('[{:}] - [get_best_arch] : {:} -> {:}'.format(epoch_str, genotype, temp_accuracy))
    valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion, xargs.algo, logger)
    logger.log('[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}% | {:}'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5, genotype))
    valid_accuracies[epoch] = valid_a_top1

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
                'valid_accuracies' : valid_accuracies},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)
    with torch.no_grad():
      logger.log('{:}'.format(search_model.show_alphas()))
    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch], '200')))
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  wandb.log({"supernet_train_time":search_time.sum})

  # the final post procedure : count the time
  start_time = time.time()

  if xargs.cand_eval_method == 'val_acc':
    genotype, temp_accuracy = get_best_arch(train_loader, valid_loader, network, xargs.eval_candidate_num, xargs.algo, logger=logger, style=xargs.cand_eval_method, api=api)
  elif xargs.cand_eval_method == 'sotl': #TODO probably get rid of this
    # if xargs.sotl_dataset_eval == 'train_val':
    #   sotl_loader = itertools.chain(train_loader, valid_loader)
    # elif xargs.sotl_dataset_eval == 'train':
    #   sotl_loader = train_loader
    # elif xargs.sotl_dataset_eval == 'val':
    #   sotl_loader = valid_loader
    genotype, temp_accuracy = get_best_arch(train_loader, valid_loader, network, xargs.eval_candidate_num, xargs.algo, logger=logger, style=xargs.cand_eval_method, 
      w_optimizer=w_optimizer, w_scheduler=w_scheduler, config=config, epochs=xargs.eval_epochs, steps_per_epoch=xargs.steps_per_epoch, 
      api=api, additional_training = xargs.additional_training, val_loss_freq=xargs.val_loss_freq, 
      overwrite_additional_training=xargs.overwrite_additional_training, scheduler_type=xargs.scheduler, xargs=xargs)

  if xargs.algo == 'setn' or xargs.algo == 'enas':
    network.set_cal_mode('dynamic', genotype)
  elif xargs.algo == 'gdas':
    network.set_cal_mode('gdas', None)
  elif xargs.algo.startswith('darts'):
    network.set_cal_mode('joint', None)
  elif xargs.algo == 'random':
    network.set_cal_mode('urs', None)
  else:
    raise ValueError('Invalid algorithm name : {:}'.format(xargs.algo))
  search_time.update(time.time() - start_time)

  valid_a_loss , valid_a_top1 , valid_a_top5 = valid_func(valid_loader, network, criterion, xargs.algo, logger)
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
  parser.add_argument('--dataset'     ,       type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--search_space',       type=str,   default='tss', choices=['tss'], help='The search space name.')
  parser.add_argument('--algo'        ,       type=str,   choices=['darts-v1', 'darts-v2', 'gdas', 'setn', 'random', 'enas'], help='The search space name.')
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
  parser.add_argument('--workers',            type=int,   default=1,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   default='./output/search', help='Folder to save checkpoints and log.')
  parser.add_argument('--print_freq',         type=int,   default=200,  help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  parser.add_argument('--cand_eval_method',          type=str,   help='SoTL or ValAcc', default='val_acc', choices = ['sotl', 'val_acc'])
  parser.add_argument('--sotl_dataset_eval',          type=str,   help='Whether to do the SoTL short training on the train+val dataset or the test set', default='train', choices = ['train_val', "train", 'test'])
  parser.add_argument('--sotl_dataset_train',          type=str,   help='TODO doesnt work currently. Whether to do the train step in SoTL on the whole train dataset (ie. the default split of CIFAR10 to train/test) or whether to use the extra split of train into train/val', 
    default='train', choices = ['train_val', 'train'])
  parser.add_argument('--steps_per_epoch',           default=100,  help='Number of minibatches to train for when evaluating candidate architectures with SoTL')
  parser.add_argument('--eval_epochs',          type=int, default=1,   help='Number of epochs to train for when evaluating candidate architectures with SoTL')
  parser.add_argument('--additional_training',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=True,   help='Whether to train the supernet samples or just go through the training loop with no grads')
  parser.add_argument('--val_batch_size',          type=int, default=None,   help='Batch size for the val loader - this is crucial for SoVL and similar experiments, but bears no importance in the standard NASBench setup')
  parser.add_argument('--dry_run',          type=bool, default=False,   help='WANDB dry run - whether to sync to the cloud')
  parser.add_argument('--val_dset_ratio',          type=float, default=1,   help='Only uses a ratio of X for the valid data loader. Used for testing SoValAcc robustness')
  parser.add_argument('--val_loss_freq',          type=int, default=1,   help='How often to calculate val loss during training. Probably better to only this for smoke tests as it is generally better to record all and then post-process if different results are desired')
  parser.add_argument('--overwrite_additional_training',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False,   help='Whether to load checkpoints of additional training')
  parser.add_argument('--scheduler',          type=str, default=None, choices=['linear', 'cos_reinit', 'cos_adjusted'],   help='Whether to use different training protocol for the postnet training')
  parser.add_argument('--train_batch_size',          type=int, default=None,   help='Training batch size for the POST-SUPERNET TRAINING!')


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
