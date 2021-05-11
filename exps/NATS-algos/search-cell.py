##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v1 --rand_seed 780 --dry_run=True --merge_train_val_supernet=True --search_batch_size=2 --supernet_init_path=1
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v1 --drop_path_rate 0.3
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v1
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v2 --rand_seed 777 
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v2
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v2
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo gdas --rand_seed 777 --merge_train_val_supernet=True
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo gdas
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo gdas
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo setn --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo setn
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo setn
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 6 --cand_eval_method sotl --search_epochs=3 --steps_per_epoch 15 --train_batch_size 16 --eval_epochs 1 --eval_candidate_num 5 --val_batch_size 32 --scheduler constant --overwrite_additional_training True --dry_run=False --individual_logs False --greedynas_epochs=3 --search_batch_size=64 --greedynas_sampling=random --inner_steps=2 --meta_algo=metaprox
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch 10 --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 64 --dry_run=True --train_batch_size 64 --val_dset_ratio 0.2
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 3 --cand_eval_method sotl --steps_per_epoch 15 --eval_epochs 1 --search_space_paper=darts --max_nodes=7 --num_cells=8
# python ./exps/NATS-algos/search-cell.py --algo=random --cand_eval_method=sotl --data_path=$TORCH_HOME/cifar.python --dataset=cifar10 --eval_epochs=2 --rand_seed=2 --steps_per_epoch=None
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo random
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch 5 --train_batch_size 128 --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 32 --scheduler cos_fast --lr 0.003 --overwrite_additional_training True --dry_run=False --reinitialize True --individual_logs False
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777

# python ./exps/NATS-algos/search-cell.py --dataset cifar5m  --data_path 'D:\' --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch 5 --train_batch_size 128 --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 32 --scheduler cos_fast --lr 0.003 --overwrite_additional_training True --dry_run=True --reinitialize True --individual_logs False --total_samples=600000
# python ./exps/NATS-algos/search-cell.py --dataset cifar5m  --data_path '$TORCH_HOME/cifar.python' --algo darts-v1 --rand_seed 774 --dry_run=True --total_samples=600000
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
  arch_percentiles, init_grad_metrics, closest_epoch, estimate_epoch_equivalents, rolling_window, nn_dist, interpolate_state_dicts, avg_state_dicts)
from utils.train_loop import (sample_new_arch, format_input_data, update_brackets)
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
import higher
import higher.patch
import higher.optim
from hessian_eigenthings import compute_hessian_eigenthings


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

def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, algo, logger, args=None, epoch=None, smoke_test=False, 
  meta_learning=False, api=None, supernets_decomposition=None, arch_groups_quartiles=None, arch_groups_brackets: Dict=None, 
  all_archs=None, grad_metrics_percentiles=None, metrics_percs=None, percentiles=None, loss_threshold=None, replay_buffer = None, checkpoint_freq=3, val_loader=None, meta_optimizer=None):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(track_std=True), AverageMeter(track_std=True), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(track_std=True), AverageMeter(track_std=True), AverageMeter()
  if arch_groups_brackets is not None:
    all_brackets = set(arch_groups_brackets.values())
  end = time.time()
  network.train()
  parsed_algo = algo.split("_")
  if args.search_space_paper == "nats-bench":
    if (len(parsed_algo) == 3 and ("perf" in algo or "size" in algo)): # Can be used with algo=random_size_highest etc. so that it gets parsed correctly
      arch_sampler = ArchSampler(api=api, model=network, mode=parsed_algo[1], prefer=parsed_algo[2])
    else:
      arch_sampler = ArchSampler(api=api, model=network, mode="perf", prefer="random") # TODO mode=perf is a placeholder so that it loads the perf_all_dict, but then we do sample(mode=random) so it does not actually exploit the perf information
  else:
    arch_sampler = None
  losses_percs = {"perc"+str(percentile): AverageMeter() for percentile in percentiles}
  supernet_train_stats = {"train_loss":{"sup"+str(percentile): [] for percentile in all_brackets}, 
    "val_loss": {"sup"+str(percentile): [] for percentile in all_brackets},
    "val_acc": {"sup"+str(percentile): [] for percentile in all_brackets},
    "train_acc": {"sup"+str(percentile): [] for percentile in all_brackets}}
  if args.search_space_paper == "nats-bench":
    supernet_train_stats_by_arch = {arch: {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []} for arch in arch_sampler.archs}
    supernet_train_stats_avgmeters = {}
  for k in list(supernet_train_stats.keys()):
    supernet_train_stats[k+str("AVG")] = {"sup"+str(percentile): [] for percentile in all_brackets}
    supernet_train_stats_avgmeters[k+str("AVG")] = {"sup"+str(percentile): AverageMeter() for percentile in all_brackets}


  grad_norm_meter, meta_grad_timer = AverageMeter(), AverageMeter() # NOTE because its placed here, it means the average will restart after every epoch!
  if args.meta_algo is not None:
    model_init = deepcopy(network)
  else:
    model_init = None
  arch_overview = {"cur_arch": None, "all_cur_archs": [], "all_archs": [], "top_archs_last_epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
  search_loader_iter = iter(xloader)
  if args.inner_steps is not None:
    inner_steps = args.inner_steps
  else:
    inner_steps = 1 # SPOS equivalent
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in tqdm(enumerate(search_loader_iter), desc = "Iterating over SearchDataset", total = round(len(xloader)/(inner_steps if not args.inner_steps_same_batch else 1))): # Accumulate gradients over backward for sandwich rule
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, args)

    if smoke_test and step >= 3:
      break
    if step == 0:
      logger.log(f"New epoch of arch; for debugging, those are the indexes of the first minibatch in epoch: {base_targets[0:10]}")
    scheduler.update(None, 1.0 * step / len(xloader))
    # measure data loading time
    data_time.update(time.time() - end)

    if (args.sandwich is None or args.sandwich == 1):
      num_iters = 1
    else:
      num_iters = args.sandwich
      if args.sandwich_computation == "parallel":
        # TODO I think this wont work now that I reshuffled the for loops around for implementing Higher
        # DataParallel should be fine and we do actually want to share the same data across all models. But would need multi-GPU setup to check it out, it does not help on Single GPU

        if epoch == 0:
          logger.log(f"Computing parallel sandwich forward pass at epoch = {epoch}")
        # Prepare the multi-path samples in advance for Parallel Sandwich
        if all_archs is not None:
          sandwich_archs = [random.sample(all_archs, 1)[0] for _ in range(args.sandwich)]
        else:
          sandwich_archs = [arch_sampler.sample(mode="random", candidate_num = 1)[0] for _ in range(args.sandwich)]
        network.zero_grad()
        network.set_cal_mode('sandwich', sandwich_cells = sandwich_archs)
        network.logits_only = True

        if args.sandwich is not None and args.sandwich > 1:
          parallel_model = nn.DataParallel(network, device_ids = [0 for _ in range(args.sandwich)])
        parallel_inputs = base_inputs.repeat((args.sandwich, 1, 1, 1))
        parallel_targets = base_targets.repeat(args.sandwich)

        all_logits = parallel_model(parallel_inputs)
        parallel_loss = criterion(all_logits, parallel_targets)/args.sandwich
        parallel_loss.backward()
        split_logits = torch.split(all_logits, base_inputs.shape[0], dim=0)

        network.logits_only = False
      elif args.sandwich_computation == "parallel_custom":
        # TODO probably useless. Does not provide any speedup due to only a single CUDA context being active on the GPU at a time even though the jobs are queued asynchronously
        network.zero_grad()
        all_logits = []
        all_base_inputs = [deepcopy(base_inputs) for _ in range(args.sandwich)]
        all_models = [deepcopy(network) for _ in range(args.sandwich)]
        for sandwich_idx, sandwich_arch in enumerate(sandwich_archs):
          cur_model = all_models[sandwich_idx]
          cur_model.set_cal_mode('dynamic', sandwich_arch)
          _, logits = cur_model(base_inputs)
          all_logits.append(logits)
        all_losses = [criterion(logits, base_targets) * (1 if args.sandwich is None else 1/args.sandwich) for logits in all_logits]
        for loss in all_losses:
          loss.backward()
        split_logits = all_logits

    inner_rollouts = [] # For implementing meta-batch_size in Reptile/MetaProx and similar
    meta_grads = []
    for outer_iter in range(num_iters):
      # Update the weights
      if args.meta_algo is not None:
        network.load_state_dict(model_init.state_dict())
        if step <= 1:
          logger.log(f"After restoring original params: Original net: {str(list(model_init.parameters())[1])[0:80]}, after-rollout net: {str(list(network.parameters())[1])[0:80]}")

      sampling_done, lowest_loss_arch, lowest_loss = False, None, 10000 # Used for GreedyNAS online search space pruning - might have to resample many times until we find an architecture below the required threshold
      while not sampling_done: # TODO the sampling_done should be useful for like online sampling with rejections maybe
        if algo == 'setn':
          sampled_arch = network.dync_genotype(True)
          network.set_cal_mode('dynamic', sampled_arch)
        elif algo == 'gdas':
          network.set_cal_mode('gdas', None)
          sampled_arch = network.genotype
        elif algo.startswith('darts'):
          network.set_cal_mode('joint', None)
          sampled_arch = network.genotype

        elif "random_" in algo and len(parsed_algo) > 1 and ("perf" in algo or "size" in algo):
          if args.search_space_paper == "nats-bench":
            sampled_arch = arch_sampler.sample()[0]
            network.set_cal_mode('dynamic', sampled_arch)
          else:
            network.set_cal_mode('urs')
        # elif "random" in algo and args.evenly_split is not None: # TODO should just sample outside of the function and pass it in as all_archs?
        #   sampled_arch = arch_sampler.sample(mode="evenly_split", candidate_num = args.eval_candidate_num)[0]
        #   network.set_cal_mode('dynamic', sampled_arch)

        elif "random" in algo and args.sandwich is not None and args.sandwich > 1 and args.sandwich_computation == "parallel":
          assert args.sandwich_mode != "quartiles", "Not implemented yet"
          sampled_arch = sandwich_archs[outer_iter]
          network.set_cal_mode('dynamic', sampled_arch)

        elif "random" in algo and args.sandwich is not None and args.sandwich > 1 and args.sandwich_mode == "quartiles":
          if args.search_space_paper == "nats-bench":
            assert args.sandwich == 4 # 4 corresponds to using quartiles
            if step == 0:
              logger.log(f"Sampling from the Sandwich branch with sandwich={args.sandwich} and sandwich_mode={args.sandwich_mode}")
            sampled_archs = arch_sampler.sample(mode = "quartiles", subset = all_archs, candidate_num=args.sandwich) # Always samples 4 new archs but then we pick the one from the right quartile
            sampled_arch = sampled_archs[outer_iter] # Pick the corresponding quartile architecture for this iteration
            network.set_cal_mode('dynamic', sampled_arch)
          else:
            network.set_cal_mode('urs')
        elif "random_" in algo and "grad" in algo:
          network.set_cal_mode('urs')
        elif algo == 'random': # NOTE the original branch needs to be last so that it is fall-through for all the special 'random' branches
          if supernets_decomposition or all_archs is not None or arch_groups_brackets is not None:
            if all_archs is not None:
              sampled_arch = random.sample(all_archs, 1)[0]
              network.set_cal_mode('dynamic', sampled_arch)
            else:
              if args.search_space_paper == "nats-bench":
                sampled_arch = arch_sampler.sample(mode="random")[0]
                network.set_cal_mode('dynamic', sampled_arch)
              else:
                network.set_cal_mode('urs', None)
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

      if args.meta_algo and args.meta_algo not in ['reptile', 'metaprox']: # NOTE first order algorithms have separate treatment because they are much sloer with Higher
        fnetwork = higher.patch.monkeypatch(network, device='cuda', copy_initial_weights=True, track_higher_grads = True if args.meta_algo not in ['reptile', 'metaprox'] else False)
        diffopt = higher.optim.get_diff_optim(w_optimizer, network.parameters(), fmodel=fnetwork, device='cuda', override=None, track_higher_grads = True) 
        fnetwork.zero_grad() # TODO where to put this zero_grad? was there below in the sandwich_computation=serial branch, tbut that is surely wrong since it wouldnt support higher meta batch size
      else: 
        fnetwork = network
        diffopt = w_optimizer

      sotl = 0
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
        if step in [0, 1] and inner_step < 3 and epoch % 5 == 0:
          logger.log(f"Base targets in the inner loop at inner_step={inner_step}, step={step}: {base_targets[0:10]}")
        if args.sandwich_computation == "serial":
          _, logits = fnetwork(base_inputs)
          base_loss = criterion(logits, base_targets) * (1 if args.sandwich is None else 1/args.sandwich)
          sotl += base_loss
        else:
          # Parallel computation branch - we have precomputed this before the for loop
          base_loss = parallel_loss
          logits = split_logits[outer_iter]
        if outer_iter == num_iters - 1 and replay_buffer is not None and args.replay_buffer > 0: # We should only do the replay once regardless of the architecture batch size
          # TODO need to implement replay support for DARTS space (in general, for cases where we do not get an arch directly but instead use uniform sampling at each choice block)
          for replay_arch in replay_buffer:
            fnetwork.set_cal_mode('dynamic', replay_arch)
            _, logits = fnetwork(base_inputs)
            replay_loss = criterion(logits, base_targets)
            if epoch in [0,1] and step == 0:
              logger.log(f"Replay loss={replay_loss.item()} for {len(replay_buffer)} items with num_iters={num_iters}, outer_iter={outer_iter}, replay_buffer={replay_buffer}") # Debugging messages
            base_loss = base_loss + (args.replay_buffer_weight / args.replay_buffer) * replay_loss # TODO should we also specifically add the L2 regularizations as separate items? Like this, it diminishes the importance of weight decay here
            fnetwork.set_cal_mode('dynamic', arch_overview["cur_arch"])
        if args.meta_algo == "metaprox":
          proximal_penalty = nn_dist(fnetwork, model_init)
          if epoch % 5 == 0 and step in [0, 1]:
            logger.log(f"Proximal penalty at epoch={epoch}, step={step} was found to be {proximal_penalty}")
          base_loss = base_loss + args.metaprox_lambda/2*proximal_penalty # TODO scale by sandwich size?
        if args.sandwich_computation == "serial": # the Parallel losses were computed before
          if (not args.meta_algo) or args.first_order_debug or args.meta_algo in ['reptile', 'metaprox']:
            base_loss.backward()

        if args.meta_algo and not args.first_order_debug and args.meta_algo not in ['reptile', 'metaprox']:
          diffopt.step(base_loss)
        elif args.meta_algo in ['reptile', 'metaprox']: # Inner loop update for first order algorithms
          w_optimizer.step()
        else:
          pass # Standard multi-path branch

        if 'gradnorm' in algo: # Normalize gradnorm so that all updates have the same norm. But does not work well at all in practice
          coef, total_norm = grad_scale(w_optimizer.param_groups[0]["params"], grad_norm_meter.avg)
          grad_norm_meter.update(total_norm)

        if supernets_decomposition is not None:
          # TODO need to fix the logging here I think. The normal logging is much better now
          cur_quartile = arch_groups_quartiles[sampled_arch.tostr()]
          with torch.no_grad():
            dw = [p.grad.detach().to('cpu') if p.grad is not None else torch.zeros_like(p).to('cpu') for p in fnetwork.parameters()]
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
        if inner_step == inner_steps -1:
          if args.meta_algo in ['reptile', 'metaprox']:
            inner_rollouts.append(deepcopy(fnetwork.state_dict()))
          elif args.meta_algo:
            if args.higher_method == "val":
              _, logits = fnetwork(arch_inputs)
              arch_loss = criterion(logits, arch_targets)
            elif args.higher_method == "sotl":
              arch_loss = sotl
            meta_grad_start = time.time()
            meta_grad = torch.autograd.grad(arch_loss, fnetwork.parameters(time=0), allow_unused=True)
            meta_grad_timer.update(time.time() - meta_grad_start)

            meta_grads.append(meta_grad)

      base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
      base_losses.update(base_loss.item() / (1 if args.sandwich is None else 1/args.sandwich),  base_inputs.size(0))
      base_top1.update  (base_prec1.item(), base_inputs.size(0))
      base_top5.update  (base_prec5.item(), base_inputs.size(0))
      arch_overview["train_acc"].append(base_prec1)
      arch_overview["train_loss"].append(base_loss.item())

      update_brackets(supernet_train_stats_by_arch, supernet_train_stats, supernet_train_stats_avgmeters, arch_groups_brackets, arch_overview, 
        [("train_loss", base_loss.item() / (1 if args.sandwich is None else 1/args.sandwich)), ("train_acc", base_prec1.item())], all_brackets, sampled_arch,  args)
      
      if all_archs is not None: # Correctness chekcs
        assert sampled_arch in all_archs 

    if args.meta_algo is None:
      # The standard multi-path branch. Note we called base_loss.backward() earlier for this meta_algo-free code branch
      w_optimizer.step()
      fnetwork.zero_grad()

    # Updating archs after all weight updates are finished
    for previously_sampled_arch in arch_overview["all_cur_archs"]:
      arch_loss = torch.tensor(10) # Placeholder in case it never gets updated here. It is not very useful in any case
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
      if algo == 'darts-v2' and not args.meta_algo:
        arch_loss, logits = backward_step_unrolled(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets, meta_learning=meta_learning)
        a_optimizer.step()
      elif (algo == 'random' or algo == 'enas' or 'random' in algo ) and not args.meta_algo:
        if algo == "random":
          arch_loss = torch.tensor(10) # Makes it slower and does not return anything useful anyways
        else:
          with torch.no_grad():
            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)

      elif args.meta_algo in ['reptile', 'metaprox']: # Do the interpolation update after all meta_batch outer iters are finished
        # NOTE this updates meta-weights only! Reptile/Metaprox have no concept of architecture.
        avg_inner_rollout = avg_state_dicts(inner_rollouts)
        if step == 0:
          for i, rollout in enumerate(inner_rollouts):
            logger.log(f"Printing {i}-th rollout's weight sample: {str(list(rollout.values())[1])[0:75]}")
          logger.log(f"Average of all rollouts: {str(list(avg_inner_rollout.values())[1])[0:75]}")
        # Prepare for the interpolation step of Reptile or MetaProx
        new_state_dict = interpolate_state_dicts(model_init.state_dict(), avg_inner_rollout, args.interp_weight)
        if step == 0 and epoch % 5 == 0:
          logger.log(f"Interpolated inner_rollouts dict after {inner_step+1} steps, example parameters (note that they might be non-active in the current arch and thus be the same across all nets!) for original net: {str(list(model_init.parameters())[1])[0:80]}, after-rollout net: {str(list(network.parameters())[1])[0:80]}, interpolated (interp_weight={args.interp_weight}) state_dict: {str(list(new_state_dict.values())[1])[0:80]}")
        network.load_state_dict(new_state_dict)
        del fnetwork # Cleanup since not using the Higher context manager currently
        del diffopt

      elif args.meta_algo: 

        avg_meta_grad = [sum(grads)/len(meta_grads) for grads in zip(*meta_grads)]
        with torch.no_grad():
          for (n,p), g in zip(network.named_parameters(), avg_meta_grad):
            cond = 'arch' not in n if args.higher_params == "weights" else 'arch' in n
            if cond:
              if g is not None and p.requires_grad:
                p.grad = g
        # w_optimizer.step()
        meta_optimizer.step()
        del fnetwork # Cleanup since not using the Higher context manager currently
        del diffopt

      elif algo == "darts-v1":
        # The Darts-V1/FOMAML branch
        _, logits = network(arch_inputs)
        arch_loss = criterion(logits, arch_targets)
        arch_loss.backward()
        a_optimizer.step()
      else:
        raise NotImplementedError # Should be using the darts-v1 branch but I do not like the fallthrough here

      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
      arch_overview["val_acc"].append(arch_prec1)
      arch_overview["val_loss"].append(arch_loss.item())

      update_brackets(supernet_train_stats_by_arch, supernet_train_stats, supernet_train_stats_avgmeters, arch_groups_brackets, arch_overview, 
        [("val_loss", arch_loss.item()), ("val_acc", arch_prec1.item())], all_brackets, sampled_arch,  args)

    if args.meta_algo is not None:
      if step <= 1:
        logger.log(f"Before reassigning model_init: Original net: {str(list(model_init.parameters())[1])[0:80]}, after-rollout net: {str(list(network.parameters())[1])[0:80]}")
      model_init = deepcopy(network) # Need to make another copy of initial state for rollout-based algorithms

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

  if args.hessian:
    network.logits_only=True
    eigenvals, eigenvecs = compute_hessian_eigenthings(network, val_loader, criterion, 1, mode="power_iter", power_iter_steps=50, max_samples=128, arch_only=True, full_dataset=False)
    dom_eigenvalue = eigenvals[0]
    network.logits_only=False
  else:
    dom_eigenvalue = None

  # Add standard deviations to metrics tracked during supernet training
  new_stats = {k:v for k, v in supernet_train_stats.items()}
  for key in supernet_train_stats.keys():
    train_stats_keys = list(supernet_train_stats[key].keys())
    for bracket in train_stats_keys:
      window = rolling_window(supernet_train_stats[key][bracket], 10)
      new_stats[key][bracket+".std"] = np.std(window, axis=-1)
  supernet_train_stats = {**supernet_train_stats, **new_stats}

  search_metric_stds = {"train_loss.std": base_losses.std, "train_loss_arch.std": base_losses.std, "train_acc.std": base_top1.std, "train_acc_arch.std": arch_top1.std}
  logger.log(f"Average gradient norm over last epoch was {grad_norm_meter.avg}, min={grad_norm_meter.min}, max={grad_norm_meter.max}")
  logger.log(f"Average meta-grad time was {meta_grad_timer.avg}")
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg, supernet_train_stats, supernet_train_stats_by_arch, arch_overview, search_metric_stds, dom_eigenvalue


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
  scheduler_type:str=None, xargs:Namespace=None, train_loader_stats=None, val_loader_stats=None, 
  model_config=None, all_archs=None, search_sotl_stats=None, checkpoint_freq=3, search_epoch=None):
  true_archs = None
  with torch.no_grad():
    network.eval()
    if 'random' in algo:
      if xargs.evenly_split is not None:
        arch_sampler = ArchSampler(api=api, model=network, mode=xargs.evenly_split, dataset = xargs.evenly_split_dset)
        archs = arch_sampler.sample(mode="evenly_split", candidate_num=xargs.eval_candidate_num)
        decision_metrics = []
      elif api is not None and xargs is not None:
        archs, decision_metrics = network.return_topK(n_samples, True, api=api, dataset=xargs.dataset, size_percentile=xargs.size_percentile, perf_percentile=xargs.perf_percentile), []
      else:
        archs, decision_metrics = network.return_topK(n_samples, True), []
    elif algo == 'setn':
      logger.log(f"Sampled {n_samples} SETN architectures using the Template network")
      archs, decision_metrics = network.return_topK(n_samples, False), []

    elif algo.startswith('darts') or algo == 'gdas':
      arch = network.genotype
      true_archs, true_decision_metrics = [arch], [] # Put the same arch there twice for the rest of the code to work in idempotent way
      archs, decision_metrics = network.return_topK(n_samples, True, api=api, dataset=xargs.dataset, size_percentile=xargs.size_percentile, perf_percentile=xargs.perf_percentile), []

    elif algo == 'enas':
      archs, decision_metrics = [], []
      for _ in range(n_samples):
        _, _, sampled_arch = network.controller()
        archs.append(sampled_arch)
    else:
      raise ValueError('Invalid algorithm name : {:}'.format(algo))
    
    if all_archs is not None: # Overwrite the just sampled archs with the ones that were supplied. Useful in order to match up with the archs used in search_func
      logger.log(f"Overwrote arch sampling in get_best_arch with a subset of len={len(all_archs)}, head = {[api.archstr2index[arch.tostr()] for arch in all_archs[0:10]]}")
      archs = all_archs
    else:
      logger.log(f"Were not supplied any limiting subset of archs so instead just sampled fresh ones with len={len(archs)}, head = {[api.archstr2index[arch.tostr()] for arch in archs[0:10]]} using algo={algo}")
    logger.log(f"Running get_best_arch (evenly_split={xargs.evenly_split}, style={style}, evenly_split_dset={xargs.evenly_split_dset}) with initial seeding of archs head:{[api.archstr2index[arch.tostr()] for arch in archs[0:10]]}")
    
    # The true rankings are used to calculate correlations later
    true_rankings, final_accs = get_true_rankings(archs, api)
    true_rankings_rounded, final_accs_rounded = get_true_rankings(archs, api, decimals=3) # np.round(0.8726, 3) gives 0.873, ie. we wound accuracies to nearest 0.1% 

    if true_archs is not None:
      true_rankings_final, final_accs_final = get_true_rankings(true_archs, api)
      assert len(true_archs) == 1
      wandb.log({"true":final_accs_final[true_archs[0]], "epoch": search_epoch}) # Log the final selected arch accuracy by GDAS/DARTS as separate log entry

    upper_bound = {}
    for n in [1,5,10]:
      upper_bound[f"top{n}"] = {"cifar10":0, "cifar10-valid":0, "cifar100":0, "ImageNet16-120":0}
      for dataset in true_rankings.keys():
        upper_bound[f"top{n}"][dataset] += sum([x["metric"] for x in true_rankings[dataset][0:n]])/min(n, len(true_rankings[dataset][0:n]))
    upper_bound = {"upper":upper_bound}
    logger.log(f"Upper bound: {upper_bound}")
    
    if steps_per_epoch is not None and steps_per_epoch != "None":
      steps_per_epoch = int(steps_per_epoch)
    elif steps_per_epoch in [None, "None"]:
      steps_per_epoch = len(train_loader)
    else:
      raise NotImplementedError
    if style in ['val_acc', 'val']:
      # Original code branch from the AutoDL repo, although slightly groomed. Still relevant for get_best_arch calls during the supernet search phase
      if len(archs) >= 1:
        corrs = {"archs": [arch.tostr() for arch in archs]}
        decision_metrics_eval = {"archs": [arch.tostr() for arch in archs]}
        search_summary_stats = {"search":defaultdict(lambda: defaultdict(dict)), "epoch": search_epoch}
        for data_type in ["val", "train"]:
          for metric in ["acc", "loss", "kl"]:
            if metric == "kl" and not ('darts' in xargs.algo):
              continue
            cur_loader = valid_loader if data_type == "val" else train_loader

            decision_metrics_computed = eval_archs_on_batch(xloader=cur_loader, archs=archs, network=network, criterion=criterion, metric=metric, 
              train_loader=train_loader, w_optimizer=w_optimizer, train_steps=xargs.eval_arch_train_steps, same_batch = True) 
            try:
              corr_per_dataset = calc_corrs_val(archs=archs, valid_accs=decision_metrics_computed, final_accs=final_accs, true_rankings=true_rankings, corr_funs=None)
              corrs["supernetcorrs_" + data_type + "_" + metric] = corr_per_dataset
            except:
              continue
            decision_metrics_eval["supernet_" + data_type + "_" + metric] = decision_metrics_computed

            search_summary_stats["search"][data_type][metric]["mean"] = np.mean(decision_metrics_computed)
            search_summary_stats["search"][data_type][metric]["std"] = np.std(decision_metrics_computed)

        decision_metrics = decision_metrics_eval["supernet_val_acc"]
        wandb.log({**corrs, **search_summary_stats})
      else:
        decision_metrics=eval_archs_on_batch(xloader=valid_loader, archs=archs, network=network, train_loader=train_loader, w_optimizer=w_optimizer, train_steps = xargs.eval_arch_train_steps)

  if style == 'sotl' or style == "sovl":
    # Branch for the single-architecture finetuning in order to collect SoTL    
    if xargs.postnet_switch_train_val:
      logger.log("Switching train and validation sets for postnet training. Useful for training on the test set if desired")
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
      try:
        checkpoint = torch.load(logger.path('corr_metrics'))
      except Exception as e:
        logger.log("Failed to load corr_metrics checkpoint, trying backup now")
        checkpoint = torch.load(os.fspath(logger.path('corr_metrics'))+"_backup")

      checkpoint_config = checkpoint["config"] if "config" in checkpoint.keys() else {}
      try:
        metrics = {k:checkpoint["metrics"][k] for k in checkpoint["metrics"].keys()}
        train_stats = checkpoint["train_stats"]
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
            if not ('eval_candidate_num' in different_items or 'evenly_split' in different_items or "perf_percentile" in different_items or "size_percentile" in different_items) and not 'darts' in algo:
              logger.log("Using the checkpoint archs as ground-truth for current run. But might be better to investigate what went wrong")
              archs = checkpoint["archs"]
              true_rankings, final_accs = get_true_rankings(archs, api)
              upper_bound = {}
              for n in [1,5,10]:
                upper_bound[f"top{n}"] = {"cifar10":0, "cifar10-valid":0, "cifar100":0, "ImageNet16-120":0}
                for dataset in true_rankings.keys():
                  upper_bound[f"top{n}"][dataset] += sum([x["metric"] for x in true_rankings[dataset][0:n]])/min(n, len(true_rankings[dataset][0:n]))
              upper_bound = {"upper":upper_bound}
            else:
              logger.log("Cannot reuse archs from checkpoint because they use different arch-picking parameters")
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
    logger.log(f"Arch ranking thresholds mapping from real-to-percentiles: {arch_rankings_thresholds_nominal}")

    if xargs.evenify_training:
      # Those two lines are just to get the proper criterion to use
      config_opt = load_config('./configs/nas-benchmark/hyper-opts/200E.config', None, logger)
      _, _, criterion = get_optim_scheduler(network.weights, config_opt)
      
      epoch_eqs = estimate_epoch_equivalents(archs=archs, network=network, api=api, criterion=criterion, train_loader=train_loader, steps=15)
      max_epoch_attained = max([x["val"] for x in epoch_eqs.values()])
      logger.log(f"Evenifying the training so that all architectures have the equivalent of {max_epoch_attained} of training measured by their own training curves")

    if xargs.adaptive_lr:
      lr_counts = defaultdict(int)

    logger.log(f"Starting finetuning at {start_arch_idx} with total len(archs)={len(archs)}")
    avg_arch_time = AverageMeter()
    for arch_idx, sampled_arch in tqdm(enumerate(archs[start_arch_idx:], start_arch_idx), desc="Iterating over sampled architectures", total = len(archs)-start_arch_idx):
      assert (all_archs is None) or (sampled_arch in all_archs), "There must be a bug since we are training an architecture that is not in the supplied subset"
      arch_start = time.time()
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
      logger.log(f"Picking the scheduler with scheduler_type={scheduler_type}, xargs.lr={xargs.lr}, xargs.postnet_decay={xargs.postnet_decay}")
      if scheduler_type in ['linear_warmup', 'linear']:
        config = config._replace(scheduler=scheduler_type, warmup=1, eta_min=0, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type == "cos_reinit":
        # In practice, this leads to constant LR = 0.025 since the original Cosine LR is annealed over 100 epochs and our training schedule is very short
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type in ['cos_adjusted']:
        config = config._replace(scheduler='cos', warmup=0, epochs=epochs, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type in ['cos_fast']:
        config = config._replace(scheduler='cos', warmup=0, LR=0.001 if xargs.lr is None else xargs.lr, epochs=epochs, eta_min=0, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type in ['cos_warmup']:
        config = config._replace(scheduler='cos', warmup=1, LR=0.001 if xargs.lr is None else xargs.lr, epochs=epochs, eta_min=0, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
      elif scheduler_type in ["scratch"]:
        config_opt = load_config('./configs/nas-benchmark/hyper-opts/200E.config', None, logger)
        config_opt = config_opt._replace(LR=0.1 if xargs.lr is None else xargs.lr, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config_opt)
      elif scheduler_type in ["scratch12E"]:
        config_opt = load_config('./configs/nas-benchmark/hyper-opts/12E.config', None, logger)
        config_opt = config_opt._replace(LR=0.1 if xargs.lr is None else xargs.lr, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config_opt)
      elif scheduler_type in ["scratch1E"]:
        config_opt = load_config('./configs/nas-benchmark/hyper-opts/01E.config', None, logger)
        config_opt = config_opt._replace(LR=0.1 if xargs.lr is None else xargs.lr, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config_opt)
      elif (xargs.lr is not None or (xargs.lr is None and bool(xargs.adaptive_lr) == True)) and scheduler_type == 'constant':
        config = config._replace(scheduler='constant', constant_lr=xargs.lr if not xargs.adaptive_lr else best_lr, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
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
      train_loss_total, train_acc_total, _ = valid_func(xloader=train_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
      if xargs.grads_analysis:
        analyze_grads(network=network2, grad_metrics=grad_metrics["total_train"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)
      if not xargs.merge_train_val_postnet:
        val_loss_total, val_acc_total, _ = valid_func(xloader=val_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
        if xargs.grads_analysis:
          analyze_grads(network=network2, grad_metrics=grad_metrics["total_val"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)
      else:
        val_loss_total, val_acc_total = train_loss_total, train_acc_total
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
            if batch_idx == 0 or not xargs.merge_train_val_postnet or xargs.postnet_switch_train_val:
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
          train_loss_total, train_acc_total, _ = valid_func(xloader=train_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
          if xargs.grads_analysis:
            analyze_grads(network=network2, grad_metrics=grad_metrics["total_train"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)  
          network2.zero_grad() 
          if not xargs.merge_train_val_postnet:
            val_loss_total, val_acc_total, _ = valid_func(xloader=val_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
          else:
            val_loss_total, val_acc_total = train_loss_total, train_acc_total
            if xargs.grads_analysis:
              analyze_grads(network=network2, grad_metrics=grad_metrics["total_val"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)
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
        if hasattr(train_loader.sampler, "reset_counter"): # Resetting counter is necessary for consistent epoch batch orders across architectures using the custom Sampler
          train_loader.sampler.counter += 1

      final_metric = None # Those final/decision metrics are not very useful apart from being a compatibility layer with how get_best_arch worked in the base repo
      if style == "sotl":
        final_metric = running["sotl"]
      elif style == "sovl":
        final_metric = running["sovl"]

      decision_metrics.append(final_metric)
      
      if arch_idx % checkpoint_freq == 0 or arch_idx == len(archs)-start_arch_idx-1:
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
    logger.log("Deepcopying metrics")
    original_metrics = deepcopy(metrics)
    logger.log("Calculating transforms of original metrics:")
    metrics_FD = {k+"FD": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=1).get_time_series(chunked=True, mode="fd") for arch in archs} for k,v in tqdm(metrics.items(), desc = "Calculating FD metrics") if k in ['val_acc', 'train_loss', 'val_loss']}
    metrics.update(metrics_FD)
    metrics_factory = {arch.tostr():[[] for _ in range(epochs)] for arch in archs}

    for arch in tqdm(search_sotl_stats.keys(), desc = "Adding stats from search to the finetuning metrics values by iterating over archs", total = len(search_sotl_stats.keys())):
      for metric in search_sotl_stats[arch].keys():
        if len(search_sotl_stats[arch][metric]) > 0 and arch in metrics[metric].keys():
          for epoch_idx in range(len(metrics[metric][arch])):
              # NOTE the search_sotl_stats should entries equal to sum of metrics in the specific epoch already
              new_vals_E1 = []
              new_vals_Einf = []
              Einf_sum = sum(search_sotl_stats[arch][metric])
              for vals in metrics[metric][arch]:
                if len(search_sotl_stats[arch][metric]) > 0:
                  E1_val = search_sotl_stats[arch][metric][-1]

                for val in vals:
                  new_vals_E1.append(val + E1_val)
                  new_vals_Einf.append(val + Einf_sum)
              # Later on, we might get like train_loss_searchE1E1 - this is like Sotl E1 + loss from last epoch of the greedy supernet training
              metrics.get(metric+"_searchE1", metrics_factory)[arch][epoch_idx].extend(new_vals_E1)
              metrics.get(metric+"_searchEinf", metrics_factory)[arch][epoch_idx].extend(new_vals_Einf)
              metrics.get(metric+"_searchE1_standalone", metrics_factory)[arch][epoch_idx].append([search_sotl_stats[arch][metric][-1] for _ in range(len(new_vals_E1))])
              metrics.get(metric+"_searchEinf_standalone", metrics_factory)[arch][epoch_idx].append([Einf_sum for _ in range(len(new_vals_Einf))])

    if epochs >= 1:
      metrics_E1 = {metric+"E1": {arch.tostr():SumOfWhatever(measurements=metrics[metric][arch.tostr()], e=1).get_time_series(chunked=True) for arch in archs} for metric,v in tqdm(metrics.items(), desc = "Calculating E1 metrics") if not metric.startswith("so") and not 'accum' in metric and not 'total' in metric and not 'standalone' in metric}
      metrics.update(metrics_E1)
      Einf_metrics = ["train_lossFD", "train_loss_pct"]
      metrics_Einf = {metric+"Einf": {arch.tostr():SumOfWhatever(measurements=metrics[metric][arch.tostr()], e=100).get_time_series(chunked=True) for arch in archs} for metric,v in tqdm(metrics.items(), desc = "Calculating Einf metrics") if metric in Einf_metrics and not metric.startswith("so") and not 'accum' in metric and not 'total' in metric}
      metrics.update(metrics_Einf)      
    # else:
    #   # We only calculate Sum-of-FD metrics in this case
    #   metrics_E1 = {metric+"E1": {arch.tostr():SumOfWhatever(measurements=metrics[metric][arch.tostr()], e=1).get_time_series(chunked=True) for arch in archs} for metric,v in tqdm(metrics.items(), desc = "Calculating E1 metrics") if "FD" in metric or "E1" in metric or "Einf" in metric} # Need to add the E1/Einf cases for supernet search stats which would otherwise not get SoTL-fied
    #   metrics.update(metrics_E1)
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
      if len(archs) > 1:
        corr, to_log = calc_corrs_after_dfs(epochs=epochs, xloader=train_loader, steps_per_epoch=steps_per_epoch, metrics_depth_dim=v, 
      final_accs = final_accs, archs=archs, true_rankings = true_rankings, prefix=k, api=api, wandb_log=False, corrs_freq = xargs.corrs_freq, constant=constant_metric)
        corrs["corrs_"+k] = corr
        to_logs.append(to_log)
      # if algo == "gdas" or algo.startswith('darts'):
      #   # We also sample some random subnetworks to evaluate correlations in GDAS/DARTS cases
      #   random_corr, random_to_log = calc_corrs_after_dfs(epochs=epochs, xloader=train_loader, steps_per_epoch=steps_per_epoch, metrics_depth_dim=v, 
      #     final_accs = final_accs_random, archs=random_archs, true_rankings = true_rankings_random, prefix=k, api=api, wandb_log=False, corrs_freq = xargs.corrs_freq, constant=constant_metric)
      #   corrs["corrs_"+ "rand_" + k] = random_corr
      #   to_logs.append(random_to_log)

    arch_ranking_inner = [{"arch":arch, "metric":metrics["total_arch_count"][arch][0][0]} for arch in metrics["total_arch_count"].keys()]
    arch_ranking_inner = sorted(arch_ranking_inner, key=lambda x: x["metric"], reverse=True)
    arch_true_rankings = {"cifar10":arch_ranking_inner, "cifar100":arch_ranking_inner,"cifar10-valid":arch_ranking_inner, "ImageNet16-120":arch_ranking_inner}
    for k in ["train_grad_accum", "train_lossE1", "sotl", "train_grad_mean_accum", "sogn"]:
      # TODO what was the point of this?
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
    # Produces some charts to WANDB so that it is easier to see the distribution of accuracy of sampled architectures
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

    if n_samples-start_arch_idx >= 0: #If there was training happening - might not be the case if we just loaded checkpoint
      # We reshape the stored train statistics so that it is a Seq[Dict[k: summary statistics across all archs for a timestep]] instead of Seq[Seq[Dict[k: train stat for a single arch]]]
      processed_train_stats = []
      all_threshold_keys = {}
      for key in train_stats[0][0].keys():
        all_threshold_keys[key] = None
        for threshold in arch_rankings_thresholds_nominal.values():
          all_threshold_keys[key+str(threshold)] = None
      for idx, stats_across_time in tqdm(enumerate(train_stats), desc="Processing train stats", total=len(train_stats)):
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
        if n_samples-start_arch_idx >= 0: #If there was training happening - might not be the case if we just loaded checkpoint
          all_data_to_log = {**all_batch_data, **{"summary":processed_train_stats[epoch_idx*steps_per_epoch+batch_idx]}} # We add the Summary nesting to prevent overwriting of the normal stats by the batch_train mean/stds
        else:
          logger.log("Skipped train summary logging!")
          all_data_to_log = all_batch_data

        all_data_to_log.update(upper_bound)

        wandb.log(all_data_to_log)

    wandb.log({"arch_perf":arch_perf_tables, "arch_perf_charts":arch_perf_charts})

  if style in ["sotl", "sovl"] and n_samples-start_arch_idx > 0 and arch_idx % checkpoint_freq == 0: # otherwise, we are just reloading the previous checkpoint so should not save again
    corr_metrics_path = save_checkpoint({"metrics":original_metrics, "corrs": corrs, "train_stats": train_stats,
      "archs":archs, "start_arch_idx":arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},
      logger.path('corr_metrics'), logger, backup=True)

    print(f"Upload to WANDB at {corr_metrics_path.absolute()}")
    try:
      wandb.save(str(corr_metrics_path.absolute()))
    except Exception as e:
      print(f"Upload to WANDB failed because {e}")

  best_idx = np.argmax(decision_metrics)
  try:
    best_arch, best_valid_acc = archs[best_idx], decision_metrics[best_idx]
  except Exception as e:
    logger.log(f"Failed to get best arch via decision_metrics due to {e}")
    logger.log(f"Decision metrics: {decision_metrics}")
    logger.log(f"Best idx: {best_idx}, length of archs: {len(archs)}")
    best_arch,best_valid_acc = archs[0], decision_metrics[0]

  if true_archs is not None: # ie. DARTS/GDAS/etc. cases which have a clearly set best arch
    return true_archs[0], best_valid_acc
  else: # Return the best arch evaluated by random sampling mostly
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
  import warnings # There are some PyTorch UserWarnings because of the gradient hacks later on
  warnings.filterwarnings("ignore", category=UserWarning)
  warnings.filterwarnings("ignore")

  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( max(int(xargs.workers), 1))
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(xargs)
  gpu_mem = torch.cuda.get_device_properties(0).total_memory

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
  if xargs.search_lr is not None:
    config = config._replace(LR = xargs.search_lr)
  if xargs.search_momentum is not None:
    config = config._replace(LR = xargs.search_momentum)

  resolved_train_batch_size, resolved_val_batch_size = xargs.train_batch_size if xargs.train_batch_size is not None else config.batch_size, xargs.val_batch_size if xargs.val_batch_size is not None else config.test_batch_size
  # NOTE probably better idea to not use train_batch_size here to not accidentally change the supernet search?
  search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', 
    (config.batch_size if xargs.search_batch_size is None else xargs.search_batch_size, config.test_batch_size), workers=xargs.workers, epochs=config.epochs + config.warmup, determinism=xargs.deterministic_loader, 
    merge_train_val = xargs.merge_train_val_supernet, merge_train_val_and_use_test = xargs.merge_train_val_and_use_test)

  train_data_postnet, valid_data_postnet, xshape_postnet, class_num_postnet = get_datasets(xargs.dataset_postnet, xargs.data_path, -1, mmap=xargs.mmap, total_samples=xargs.total_samples)
  search_loader_postnet, train_loader_postnet, valid_loader_postnet = get_nas_search_loaders(train_data_postnet, valid_data_postnet, xargs.dataset_postnet, 'configs/nas-benchmark/', 
    (resolved_train_batch_size, resolved_val_batch_size), workers=xargs.workers, valid_ratio=xargs.val_dset_ratio, determinism=xargs.deterministic_loader, 
    meta_learning=xargs.meta_learning, epochs=xargs.eval_epochs, merge_train_val=xargs.merge_train_val_postnet, merge_train_val_and_use_test = xargs.merge_train_val_and_use_test)
  _, train_loader_stats, val_loader_stats = get_nas_search_loaders(train_data_postnet, valid_data_postnet, xargs.dataset_postnet, 'configs/nas-benchmark/', 
    (128 if gpu_mem < 8147483648 else 1024, 128 if gpu_mem < 8147483648 else 1024), workers=xargs.workers, valid_ratio=xargs.val_dset_ratio, determinism="all", 
    meta_learning=xargs.meta_learning, epochs=xargs.eval_epochs, merge_train_val=xargs.merge_train_val_postnet, merge_train_val_and_use_test = xargs. merge_train_val_and_use_test)
  logger.log(f"Using train batch size: {resolved_train_batch_size}, val batch size: {resolved_val_batch_size}")
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces(xargs.search_space, xargs.search_space_paper)

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
  if xargs.higher_params == "weights":
    if xargs.meta_optim == "adam":
      meta_optimizer = torch.optim.Adam(search_model.weights, lr=xargs.meta_lr, betas=(0.5, 0.999), weight_decay=xargs.meta_weight_decay, eps=xargs.arch_eps)
    elif xargs.meta_optim == "sgd":
      meta_optimizer = torch.optim.SGD(search_model.weights, lr=xargs.meta_lr, momentum = xargs.meta_momentum, weight_decay = xargs.meta_weight_decay)
    elif xargs.meta_optim == "arch":
      meta_optimizer = a_optimizer
    else:
      raise NotImplementedError
    logger.log(f"Initialized meta optimizer {meta_optimizer} since higher_params={xargs.higher_params}")

  else:
    assert xargs.algo != "random"
    logger.log("Using the arch_optimizer as default when optimizing architecture with 'meta-grads' - meta_optimizer does not make sense in this case")
    meta_optimizer = a_optimizer

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
  arch_sampler = ArchSampler(api=api, model=network, mode=xargs.evenly_split, dataset=xargs.evenly_split_dset)
  messed_up_checkpoint, greedynas_archs, baseline_search_logs = False, None, None

  if xargs.supernet_init_path is not None and not last_info_orig.exists():
    whole_path = xargs.supernet_init_path
    if os.path.exists(xargs.supernet_init_path):
      pass
    else:
      try:
        seed_num = int(xargs.supernet_init_path)

        whole_path = f'./output/search-tss/cifar10/random-affine0_BN0-None/checkpoint/seed-{seed_num}-basic.pth'
      except Exception as e:
        logger.log(f"Supernet init path does not seem to be formatted as seed number - it is {xargs.supernet_init_path}, error was {e}")
    
    logger.log(f'Was given supernet checkpoint to use as initialization at {xargs.supernet_init_path}, decoded into {whole_path}')
    checkpoint = torch.load(whole_path)
    # The remaining things that are usually contained in a checkpoint are restarted to empty a bit further down
    search_model.load_state_dict(checkpoint['search_model'])

  elif last_info_orig.exists() and not xargs.reinitialize and not xargs.force_overwrite: # automatically resume from previous checkpoint
    try:
      # NOTE this code branch is replicated again further down the line so any changes need to be done to both
      logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info_orig))
      if os.name == 'nt': # The last-info pickles have PosixPaths serialized in them, hence they cannot be instantied on Windows
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
      try:
        last_info   = torch.load(last_info_orig.resolve())
        checkpoint  = torch.load(last_info['last_checkpoint'])
      except Exception as e:
        logger.log("Failed to load checkpoints due to {e} but will try to load backups now")
        try:
          last_info   = torch.load(os.fspath(last_info_orig)+"_backup")
          checkpoint  = torch.load(os.fspath(last_info['last_checkpoint'])+"_backup") 
        except Exception as e:
          logger.log(f"Failed to load checkpoint backups at last_info: {os.fspath(last_info_orig)+'_backup'}, checkpoint: {os.fspath(last_info['last_checkpoint'])+'_backup'}")
      start_epoch = last_info['epoch']
      genotypes   = checkpoint['genotypes']
      baseline  = checkpoint['baseline']
      try:
        all_search_logs = checkpoint["search_logs"]
        search_sotl_stats = checkpoint["search_sotl_stats"]
        greedynas_archs = checkpoint["greedynas_archs"]
      except Exception as e:
        all_search_logs = []
        search_sotl_stats = {arch: {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []} for arch in arch_sampler.archs}
        greedynas_archs = None
        print(f"Didnt find all_search_logs; exception was {e}")
      valid_accuracies = checkpoint['valid_accuracies']
      search_model.load_state_dict( checkpoint['search_model'] )
      w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
      w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
      a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
      logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    except Exception as e:
      logger.log(f"Checkpoint got messed up and cannot be loaded due to {e}! Will have to restart")
      messed_up_checkpoint = True

  if not (last_info_orig.exists() and not xargs.reinitialize and not xargs.force_overwrite) or messed_up_checkpoint or (xargs.supernet_init_path is not None and not last_info_orig.exists()):
    logger.log(f"""=> do not find the last-info file (or was given a checkpoint as initialization): {last_info_orig}, whose existence status is {last_info_orig.exists()}. Also, reinitialize={xargs.reinitialize}, 
      force_overwrite={xargs.force_overwrite}, messed_up_checkpoint={messed_up_checkpoint}, supernet_init_path={xargs.supernet_init_path}""")
    start_epoch, valid_accuracies, genotypes, all_search_logs, search_sotl_stats = 0, {'best': -1}, {-1: network.return_topK(1, True)[0]}, [], {arch: {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []} for arch in arch_sampler.archs}
    baseline = None
  
  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup if xargs.search_epochs is None else xargs.search_epochs
  # We simulate reinitialization by not training (+ not loading the saved state_dict earlier)
  if start_epoch > total_epoch: # In case we train for 500 epochs but then the default value for search epochs is only 100
    start_epoch = total_epoch

  if start_epoch >= total_epoch - 1 and xargs.greedynas_epochs is not None and xargs.greedynas_epochs > 0 and not xargs.overwrite_supernet_finetuning:
    # Need to restart the LR schedulers
    logger = prepare_logger(xargs, path_suffix="greedy")
    logger.log(f"Start of GreedyNAS training at epoch={start_epoch}! Will train for {xargs.greedynas_epochs} epochs more. Looking for the ckpt at {logger.path('info')}")
    config_greedynas = deepcopy(config)._replace(LR = xargs.greedynas_lr, epochs = xargs.greedynas_epochs)
    logger.log(f"GreedyNAS config: {config_greedynas}")
    w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.weights, config_greedynas)
    logger.log(f"W_optimizer: {w_optimizer}")
    logger.log(f"W_scheduler: {w_scheduler}")

    last_info_orig, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
    if last_info_orig.exists() and not xargs.reinitialize and not xargs.force_overwrite: # automatically resume from previous checkpoint
      baseline_search_logs = all_search_logs # Search logs from the checkpoint we loaded previously
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
      try:
        all_search_logs = checkpoint["search_logs"]
        search_sotl_stats = checkpoint["search_sotl_stats"]
        greedynas_archs = checkpoint["greedynas_archs"]
      except Exception as e:
        all_search_logs = []
        search_sotl_stats = {arch: {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []} for arch in arch_sampler.archs}
        greedynas_archs = None
        print(f"Failed loading checkpoint due to {e}")
      search_model.load_state_dict( checkpoint['search_model'] )
      w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
      w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
      a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
      logger.log("=> loading extra checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
      if greedynas_archs is not None:
        logger.log(f"Loaded GreedyNAS archs TODO")
    else:
      logger.log(f"Failed to find checkpoint at {last_info_orig}")
      baseline_search_logs = None

  if xargs.search_space_paper == "nats-bench":
    arch_groups_brackets =  arch_percentiles(percentiles=[0,10,20,30,40,50,60,70,80,90,100], mode="perf")
  else:
    arch_groups_brackets = None
  if arch_groups_brackets is not None and len(arch_groups_brackets) == 0: 
    print(f"Arch all dict must have gotten corrupted since arch_groups_backets has len=0. Need to re-generate them.")
    # values for _percentile are just placeholder so that the dicts get generated again
    network.generate_arch_all_dicts(api=api, perf_percentile = 0.9)

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
    if type(greedynas_archs) is list and len(greedynas_archs) == 0:
      logger.log(f"Something went wrong with GreedyNAS archs! It is equal to {greedynas_archs}")
      greedynas_archs = None
    if greedynas_archs is not None:
      pass # Must have loaded greedynas_archs from the checkpoint
    else:
      if xargs.evenly_split is not None:
        greedynas_archs = arch_sampler.sample(mode="evenly_split", candidate_num = xargs.eval_candidate_num)
        logger.log(f"GreedyNAS archs are sampled according to evenly_split={xargs.evenly_split}, candidate_num={xargs.eval_candidate_num}")
      elif xargs.greedynas_sampling == "random" or xargs.greedynas_sampling is None:
        greedynas_archs = network.return_topK(xargs.eval_candidate_num, use_random=True)
        logger.log(f"GreedyNAS archs are sampled randomly (candidate_num={xargs.eval_candidate_num}), head = {[api.archstr2index[arch.tostr()] for arch in greedynas_archs[0:10]]}")
      else:
        candidate_archs = network.return_topK(xargs.greedynas_candidate_num, use_random=True)
        if xargs.greedynas_sampling_loader == "train":
          cur_loader = train_loader_stats
        elif xargs.greedynas_sampling_loader == "val":
          cur_loader = val_loader_stats
        evaled_metrics = eval_archs_on_batch(xloader=cur_loader, archs = candidate_archs, network=network, criterion=criterion, same_batch=True, metric=xargs.greedynas_sampling, train_steps=xargs.eval_archs_train_steps, train_loader=train_loader, w_optimizer=w_optimizer)
        best_archs = sorted(list(zip(candidate_archs, evaled_metrics)), key = lambda x: x[1]) # All metrics should be so that higher is better, and we sort in ascending (ie. best is last)
        logger.log(f"GreedyNAS archs are sampled greedily (candidate_num={xargs.eval_candidate_num}), head (arch_idx, metric)={[(api.archstr2index[arch_tuple[0].tostr()], arch_tuple[1]) for arch_tuple in best_archs[-10:]]}")
        greedynas_archs = [x[0] for x in best_archs[-xargs.eval_candidate_num:]]

    logger.log(f"Sampling architectures that will be used for GreedyNAS Supernet post-main-supernet training in search_func, head = {[api.archstr2index[x.tostr()] for x in greedynas_archs[0:10]]}")
  else:
    greedynas_archs = None
  supernet_key = "supernet"
  arch_perf_percs = {k:None for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
  replay_buffer = None
  valid_a_loss , valid_a_top1 , valid_a_top5 = 0, 0, 0 # Initialization because we do not store the losses in checkpoints
  for epoch in range(start_epoch if not xargs.reinitialize else 0, total_epoch + (xargs.greedynas_epochs if xargs.greedynas_epochs is not None else 0) if not xargs.reinitialize else 0):
    if epoch == total_epoch:
      # Need to switch from nrormal supernet config to GreedyNAS config
      logger = prepare_logger(xargs, path_suffix="greedy")
      logger.log(f"Start of GreedyNAS training at epoch={start_epoch} as subsequent to normal supernet training! Will train for {xargs.greedynas_epochs} epochs more")
      config_greedynas = deepcopy(config)._replace(LR = xargs.greedynas_lr, epochs = xargs.greedynas_epochs)
      logger.log(f"GreedyNAS config: {config_greedynas}")
      w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.weights, config_greedynas)
      logger.log(f"W_optimizer: {w_optimizer}")
      logger.log(f"W_scheduler: {w_scheduler}")
    
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
    search_w_loss, search_w_top1, search_w_top5, search_a_loss, search_a_top1, search_a_top5, supernet_metrics, supernet_metrics_by_arch, arch_overview, supernet_stds, dom_eigenvalue \
                = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, xargs.algo, logger, 
                  smoke_test=xargs.dry_run, meta_learning=xargs.meta_learning, api=api, epoch=epoch,
                  supernets_decomposition=supernets_decomposition, arch_groups_quartiles=arch_groups_quartiles, arch_groups_brackets=arch_groups_brackets,
                  all_archs=archs_to_sample_from, grad_metrics_percentiles=grad_metrics_percs, 
                  percentiles=percentiles, metrics_percs=metrics_percs, args=xargs, replay_buffer=replay_buffer, val_loader=valid_loader_postnet, meta_optimizer=meta_optimizer)
    if xargs.search_space_paper == "nats-bench":
      for arch in supernet_metrics_by_arch:
        for key in supernet_metrics_by_arch[arch]:
          try:
            search_sotl_stats[arch][key].append(sum(supernet_metrics_by_arch[arch][key])/max(len(supernet_metrics_by_arch[arch][key]), 1))
          except Exception as e:
            print(e)
            print(supernet_metrics_by_arch[arch][key])


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

    if epoch % xargs.search_eval_freq == 0 or epoch == total_epoch - 1 or epoch == total_epoch or len(genotypes) == 0 or not 'random' in xargs.algo:
      genotype, temp_accuracy = get_best_arch(train_loader, valid_loader, network, xargs.eval_candidate_num, xargs.algo, xargs=xargs, criterion=criterion, logger=logger, api=api, search_epoch=epoch)
      logger.log('[{:}] - [get_best_arch] : {:} -> {:}'.format(epoch_str, genotype, temp_accuracy))
      valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion, xargs.algo, logger, steps=500 if xargs.dataset=="cifar5m" else None)
      logger.log('[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}% | {:}'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5, genotype))
    elif len(genotypes) > 0:
      genotype = genotypes[-1]
      temp_accuracy = 0
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

    valid_accuracies[epoch] = valid_a_top1

    if hasattr(search_loader.sampler, "reset_counter"):
      search_loader.sampler.counter += 1

    genotypes[epoch] = genotype

    with torch.no_grad():
      logger.log('{:}'.format(search_model.show_alphas()))
    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch], '200')))

    if xargs.supernets_decomposition:
      interim = {supernet_key+"_" + key:{} for key in grad_log_keys}
      for percentile in percentiles[1:]:
        for key in grad_log_keys:
          interim[supernet_key+"_"+key]["perc"+str(percentile)] = metrics_percs[supernet_key+"_"+key]["perc"+str(percentile)][epoch][-1] # NOTE the last list should have only one item regardless
      decomposition_logs = interim

      grad_metrics_percs["grad_accum_singleE"] = None
      grad_metrics_percs["grad_accum_singleE_tensor"] = None
    else:
      decomposition_logs = {}

    per_epoch_to_log = {"search":{"train_loss":search_w_loss,  "train_loss_arch":search_a_loss, "train_acc":search_w_top1, "train_acc_arch":search_a_top1, "epoch":epoch, "dom_eigenval":dom_eigenvalue, **supernet_stds,
      "final": summarize_results_by_dataset(genotypes[epoch], api=api, iepoch=199, hp='200')}}
    search_to_log = per_epoch_to_log
    try:
      interim = {}
      for batch_idx in range(len(search_loader)):
        interim = {}
        for metric in supernet_metrics.keys():
          for bracket in supernet_metrics[metric].keys():
            interim[metric+"."+bracket] = supernet_metrics[metric][bracket][batch_idx]

        search_to_log = {**search_to_log, **interim, "epoch":epoch, "batch":batch_idx, "true_step":epoch*len(search_loader)+batch_idx, **decomposition_logs}
        all_search_logs.append(search_to_log)
    except Exception as e:
      logger.log(f"""Failed to log per-bracket supernet searchs stats due to {e} at batch_idx={batch_idx}, metric={metric}, bracket={bracket},
         length of the supernet_metrics[metric][bracket] = {len(supernet_metrics[metric][bracket]) if bracket in supernet_metrics[metric] else 'bracket missing!'}""")
      all_search_logs.append(search_to_log)

    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    if epoch % xargs.checkpoint_freq == 0:
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
                  "archs_subset" : archs_subset,
                  "search_logs" : all_search_logs,
                  "search_sotl_stats": search_sotl_stats,
                  "greedynas_archs": greedynas_archs},
                  model_base_path, logger)
      last_info = save_checkpoint({
            'epoch': epoch + 1,
            'args' : deepcopy(args),
            'last_checkpoint': save_path,
          }, logger.path('info'), logger)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  if baseline_search_logs is not None:
    for search_log in tqdm(baseline_search_logs, desc = "Logging supernet search logs from the pretrained checkpoint"):
      wandb.log(search_log)
  else:
    logger.log("There are no pretrained search logs (in the sense that the supernet search would be initialized from a checkpoint)! Not logging anything")

  for search_log in tqdm(all_search_logs, desc = "Logging supernet search logs"):
    wandb.log(search_log)
  
  wandb.log({"supernet_train_time":search_time.sum})

  # the final post procedure : count the time
  start_time = time.time()

  if xargs.cand_eval_method in ['val_acc', 'val']:
    genotype, temp_accuracy = get_best_arch(train_loader_postnet, valid_loader_postnet, network, xargs.eval_candidate_num, xargs.algo, xargs=xargs, criterion=criterion, logger=logger, style=xargs.cand_eval_method, api=api, search_epoch=epoch)
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
      model_config=model_config, all_archs=archs_to_sample_from, search_sotl_stats = search_sotl_stats)

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
  parser.add_argument('--postnet_decay',          type=float, default=None,   help='Weight decay for the POST-SUPERNET TRAINING!')

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
  parser.add_argument('--sandwich_computation',          type=str, default="serial", choices=["serial", "parallel"], help='Do a quick search for best LR before post-supernet training')

  parser.add_argument('--force_overwrite',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Load saved seed or not')
  parser.add_argument('--greedynas_epochs',          type=int, default=None, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--greedynas_lr',          type=float, default=0.01, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--greedynas_sampling',          type=str, default="random", choices=["random", "acc", "loss"], help='Metric to sample the GreedyNAS architectures for supernet finetuning')
  parser.add_argument('--greedynas_sampling_loader',          type=str, default="train", choices=["train", "val"], help='The dataset to evaluate GreedyNAS archs on')
  parser.add_argument('--greedynas_candidate_num',          type=int, default=1000, help='The number of cand archs to evaluate for picking the best ones in GreedyNAS sampling')

  parser.add_argument('--merge_train_val_postnet',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--merge_train_val_supernet',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--postnet_switch_train_val',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--dataset_postnet',          type=str, default=None, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'cifar5m'], help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--reptile',          type=int, default=None, help='How many steps to do in Reptile rollout')
  parser.add_argument('--interp_weight',          type=float, default=0.7, help='Interpolation coefficient for Reptile')
  parser.add_argument('--replay_buffer',          type=int, default=None, help='Replay buffer to tackle multi-model forgetting')
  parser.add_argument('--replay_buffer_mode',          type=str, default="random", choices=["random", "perf", "size", None], help='How to figure out what to put in the replay buffer')
  parser.add_argument('--replay_buffer_percentile',          type=float, default=0.9, help='Replay buffer percentile of performance etc.')
  parser.add_argument('--replay_buffer_weight',          type=float, default=0.5, help='Trade off between new arch loss and buffer loss')
  parser.add_argument('--replay_buffer_metric',          type=str, default="train_loss", choices=["train_loss", "train_acc", "val_acc", "val_loss"], help='Trade off between new arch loss and buffer loss')
  parser.add_argument('--evenly_split',          type=str, default=None, choices=["perf", "size"], help='Whether to split the NASBench archs into eval_candidate_num brackets and then take an arch from each bracket to ensure they are not too similar')
  parser.add_argument('--evenly_split_dset',          type=str, default="cifar10", choices=["all", "cifar10", "cifar100", "ImageNet16-120"], help='Whether to split the NASBench archs into eval_candidate_num brackets and then take an arch from each bracket to ensure they are not too similar')
  parser.add_argument('--merge_train_val_and_use_test',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=False, help='Merges CIFAR10 train/val into one (ie. not split in half) AND then also treats test set as validation')
  parser.add_argument('--search_batch_size',          type=int, default=None, help='Controls batch size for the supernet training (search/GreedyNAS finetune phase)')
  parser.add_argument('--search_eval_freq',          type=int, default=5, help='How often to run get_best_arch during supernet training')
  parser.add_argument('--search_lr',          type=float, default=None, help='LR for teh superneat search training')
  parser.add_argument('--search_momentum',          type=float, default=None, help='Momentum in the supernet search training')
  parser.add_argument('--overwrite_supernet_finetuning',          type=lambda x: False if x in ["False", "false", "", "None"] else True, default=True, help='Whether to load additional checkpoints on top of the normal training -')
  parser.add_argument('--eval_arch_train_steps',          type=int, default=None, help='Whether to load additional checkpoints on top of the normal training -')
  parser.add_argument('--supernet_init_path' ,       type=str,   default=None, help='The path of pretrained checkpoint')
  parser.add_argument('--metaprox' ,       type=int,   default=None, help='Number of adaptation steps in MetaProx')
  parser.add_argument('--metaprox_lambda' ,       type=float,   default=0.1, help='Number of adaptation steps in MetaProx')
  parser.add_argument('--search_space_paper' ,       type=str,   default="nats-bench", choices=["darts", "nats-bench"], help='Number of adaptation steps in MetaProx')
  parser.add_argument('--checkpoint_freq' ,       type=int,   default=4, help='How often to pickle checkpoints')
  parser.add_argument('--higher_method' ,       type=str, choices=['val', 'sotl'],   default='val', help='Whether to take meta gradients with respect to SoTL or val set (which might be the same as training set if they were merged)')
  parser.add_argument('--higher_params' ,       type=str, choices=['weights', 'arch'],   default='weights', help='Whether to do meta-gradients with respect to the meta-weights or architecture')
  parser.add_argument('--meta_algo' ,       type=str, choices=['reptile', 'metaprox', 'darts_higher'],   default=None, help='Whether to do meta-gradients with respect to the meta-weights or architecture')
  parser.add_argument('--inner_steps' ,       type=int,   default=None, help='Number of steps to do in the inner loop of bilevel meta-learning')
  parser.add_argument('--inner_steps_same_batch' ,       type=lambda x: False if x in ["False", "false", "", "None"] else True,   default=True, help='Number of steps to do in the inner loop of bilevel meta-learning')
  parser.add_argument('--hessian' ,       type=lambda x: False if x in ["False", "false", "", "None"] else True,   default=False, help='Whether to track eigenspectrum in DARTS')
  parser.add_argument('--meta_optim' ,       type=str,   default="sgd", choices=['sgd', 'adam', 'arch'], help='Kind of meta optimizer')
  parser.add_argument('--meta_lr' ,       type=float,   default=0.01, help='Meta optimizer LR')
  parser.add_argument('--meta_momentum' ,       type=float,   default=0.9, help='Meta optimizer SGD momentum (if applicable)')
  parser.add_argument('--meta_weight_decay' ,       type=float,   default=5e-4, help='Meta optimizer SGD momentum (if applicable)')
  parser.add_argument('--first_order_debug' ,       type=lambda x: False if x in ["False", "false", "", "None"] else True,   default=False, help='Meta optimizer SGD momentum (if applicable)')

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
