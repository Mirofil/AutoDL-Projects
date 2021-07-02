##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts_higher --rand_seed 757 --dry_run=False --merge_train_val_supernet=True --search_batch_size=64 --higher_params=arch --higher_order=second --meta_algo=darts_higher --higher_loop=bilevel --higher_method=val_multiple --inner_steps_same_batch=False --inner_steps=100 --higher_reduction=sum --higher_reduction_outer=sum
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts_higher --rand_seed 772 --dry_run=False --merge_train_val_supernet=False --search_batch_size=64 --higher_params=arch --higher_order=first --implicit_algo=neumann --implicit_steps=10 --implicit_grad_clip=None --higher_loop=bilevel --higher_method=sotl --inner_steps_same_batch=False --inner_steps=8 --search_lr=0.001
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v1 --drop_path_rate 0.3
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path '$TORCH_HOME/cifar.python/ImageNet16' --algo darts-v1 --rand_seed 780 --dry_run=True --merge_train_val_supernet=True --search_batch_size=2
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v2 --rand_seed 777 --merge_train_val_supernet=True
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v2
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v2
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo gdas --rand_seed 777 --merge_train_val_supernet=True
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo gdas_higher --rand_seed 777 --merge_train_val_supernet=True --meta_algo=gdas_higher --higher_params=arch --higher_order=first --higher_loop=joint --inner_steps_same_batch=False --inner_steps=5 --supernet_init_path=cifar10_random_30 --search_epochs=20
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo gdas_higher --rand_seed 781 --dry_run=False --merge_train_val_supernet=True --search_batch_size=64 --higher_params=arch --higher_order=first --higher_loop=bilevel --higher_method=val_multiple --meta_algo=gdas_higher --inner_steps_same_batch=False --inner_steps=3 --steps_per_epoch_supernet=25
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo gdas
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo setn --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo setn
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo setn
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 999989 --cand_eval_method sotl --search_epochs=100 --steps_per_epoch 105 --steps_per_epoch=10 --train_batch_size 64 --eval_epochs 1 --eval_candidate_num 3 --val_batch_size 32 --scheduler constant --overwrite_additional_training True --force_overwrite=True --dry_run=False --individual_logs False --search_batch_size=64 --meta_algo=reptile_higher --inner_steps=1 --higher_method=sotl --higher_loop=bilevel --higher_params=weights --higher_order=first
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 999999 --cand_eval_method sotl --search_epochs=100 --steps_per_epoch 105 --steps_per_epoch=10 --train_batch_size 64 --eval_epochs 1 --eval_candidate_num 3 --val_batch_size 32 --scheduler constant --overwrite_additional_training True --force_overwrite=True --dry_run=False --individual_logs False --search_batch_size=64 --meta_algo=metaprox --inner_steps=3 --higher_method=sotl --higher_loop=joint --higher_params=weights
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 11000 --cand_eval_method sotl --search_epochs=1 --train_batch_size 64 --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 32 --scheduler constant --overwrite_additional_training True --dry_run=False --individual_logs False --search_batch_size=64 --greedynas_sampling=random --finetune_search=uniform --lr=0.001 --merge_train_val_supernet=True --val_dset_ratio=0.9 --force_overwrite=True
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v1 --rand_seed 4000 --cand_eval_method sotl --steps_per_epoch 15 --eval_epochs 1 --search_space_paper=darts --max_nodes=7 --num_cells=2 --search_batch_size=32 --model_name=DARTS --steps_per_epoch_supernet=5
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 1 --cand_eval_method sotl --search_epochs=100 --train_batch_size 64 --eval_epochs 1 --eval_candidate_num 100 --val_batch_size 64 --scheduler constant --dry_run=False --individual_logs False --search_batch_size=64 --finetune_search=uniform --lr=0.001 --force_overwrite=True --grad_drop_p=0.5
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 1000 --cand_eval_method sotl --eval_epochs 1 --search_space_paper=darts --max_nodes=7 --num_cells=2 --search_batch_size=64 --model_name=generic_nasnet --eval_candidate_num=350 --search_epochs=1 --steps_per_epoch=120
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 1000 --cand_eval_method sotl --eval_epochs 1 --search_space_paper=darts --max_nodes=7 --num_cells=2 --search_batch_size=64 --model_name=generic_nasnet --eval_candidate_num=350 --search_epochs=1 --steps_per_epoch=120 --sandwich=8 --sandwich_mode=fairnas
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 999999 --cand_eval_method sotl --search_epochs=100 --steps_per_epoch 105 --steps_per_epoch=10 --train_batch_size 64 --eval_epochs 1 --eval_candidate_num 3 --val_batch_size 32 --scheduler constant --overwrite_additional_training True --force_overwrite=True --dry_run=False --individual_logs False --search_batch_size=64 --meta_algo=maml_higher --inner_steps=3 --inner_steps_same_batch=True --higher_method=sotl --higher_loop=bilevel --higher_order=second --higher_params=weights
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random --rand_seed 3000 --cand_eval_method sotl --eval_epochs 1 --search_space_paper=nb101_1 --search_batch_size=64 --eval_candidate_num=200 --search_epochs=1 --steps_per_epoch=120 --save_archs_split=archs_nb101_1_random_200_seed3000.pkl

# python ./exps/NATS-algos/search-cell.py --algo=random --cand_eval_method=sotl --data_path=$TORCH_HOME/cifar.python --dataset=cifar10 --eval_epochs=2 --rand_seed=2 --steps_per_epoch=None
# python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo random
# python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo random --rand_seed 1 --cand_eval_method sotl --steps_per_epoch 5 --train_batch_size 128 --eval_epochs 1 --eval_candidate_num 2 --val_batch_size 32 --scheduler cos_fast --lr 0.003 --overwrite_additional_training True --dry_run=False --reinitialize True --individual_logs False
####
# python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777
# python ./exps/NATS-algos/search-cell.py --dataset cifar10 --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001 --rand_seed 777 --discrete_diffnas_method=val --discrete_diffnas_steps=5
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
import collections
import torch
import torch.nn as nn
import pickle
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
  arch_percentiles, init_grad_metrics, closest_epoch, estimate_epoch_equivalents, rolling_window, nn_dist, 
  interpolate_state_dicts, avg_state_dicts, _hessian, avg_nested_dict, mutate_topology_func, takespread)
from utils.train_loop import (sample_new_arch, format_input_data, update_brackets, get_finetune_scheduler, find_best_lr, 
                              sample_arch_and_set_mode, valid_func, train_controller, 
                              regularized_evolution_ws, train_epoch, evenify_training, 
                              exact_hessian, approx_hessian, backward_step_unrolled, backward_step_unrolled_darts, sample_arch_and_set_mode_search, 
                              update_supernets_decomposition, bracket_tracking_setup, update_running, update_base_metrics,
                              load_my_state_dict, resolve_higher_conds, init_search_from_checkpoint, init_supernets_decomposition,
                              scheduler_step, count_ops, grad_drop)
from utils.higher_loop import hypergrad_outer, fo_grad_if_possible, hyper_meta_step
from models.cell_searchs.generic_model import ArchSampler
from log_utils import Logger
from utils.implicit_grad import implicit_step
import wandb
import itertools
import time
import seaborn as sns
import bisect
sns.set_theme(style="whitegrid")

from argparse import Namespace
from typing import *
from tqdm import tqdm
try:
  import multiprocess as mp
except:
  import multiprocessing as mp
from utils.wandb_utils import train_stats_reporter
import higher
import higher.patch
import higher.optim
from hessian_eigenthings import compute_hessian_eigenthings

if os.environ.get("TORCH_HOME", None) is None:
  os.environ["TORCH_HOME"] = '/storage/.torch'

def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, algo, logger, xargs=None, epoch=None, smoke_test=False, 
  api=None, supernets_decomposition=None, arch_groups_quartiles=None, arch_groups_brackets: Dict=None, 
  all_archs=None, grad_metrics_percentiles=None, metrics_percs=None, percentiles=None, loss_threshold=None, replay_buffer = None, 
  checkpoint_freq=3, val_loader=None, train_loader=None, meta_optimizer=None):
  
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(track_std=True), AverageMeter(track_std=True), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(track_std=True), AverageMeter(track_std=True), AverageMeter()
  end = time.time()
  network.train()
  parsed_algo = algo.split("_")
  if xargs.search_space_paper == "nats-bench":
    if (len(parsed_algo) == 3 and ("perf" in algo or "size" in algo)): # Can be used with algo=random_size_highest etc. so that it gets parsed correctly
      arch_sampler = ArchSampler(api=api, model=network, mode=parsed_algo[1], prefer=parsed_algo[2], op_names=network._op_names, max_nodes = xargs.max_nodes, search_space = xargs.search_space_paper)
    else:
      arch_sampler = ArchSampler(api=api, model=network, mode="perf", prefer="random", op_names=network._op_names, max_nodes = xargs.max_nodes, search_space = xargs.search_space_paper) # TODO mode=perf is a placeholder so that it loads the perf_all_dict, but then we do sample(mode=random) so it does not actually exploit the perf information
  else:
    arch_sampler = None
    
  losses_percs = {"perc"+str(percentile): AverageMeter() for percentile in percentiles}
  brackets_cond = xargs.search_space_paper == "nats-bench" and arch_groups_brackets is not None
  all_brackets, supernet_train_stats, supernet_train_stats_by_arch, supernet_train_stats_avgmeters = bracket_tracking_setup(arch_groups_brackets, brackets_cond, arch_sampler)

  grad_norm_meter, meta_grad_timer = AverageMeter(), AverageMeter() # NOTE because its placed here, it means the average will restart after every epoch!
  if xargs.meta_algo is not None:
    model_init = deepcopy(network)
    logger.log(f"Initial network state at the start of search func: Original net: {str(list(model_init.parameters())[1])[0:80]}")
    logger.log(f"Arch check at start of search: Original net: {str(list(network.alphas))[0:80]}")
    orig_w_optimizer = w_optimizer
    w_optim_init = deepcopy(w_optimizer) # TODO what to do about the optimizer stes?
    if xargs.meta_algo in ['reptile', 'metaprox']:
      if not (xargs.inner_steps_same_batch is True):
        logger.log(f"WARNING! Using Reptile/Metaprox/.. and inner_step_same_batch={xargs.inner_steps_same_batch}")
      if args.sandwich is not None and args.sandwich > 1: # TODO We def dont want to be sharing Momentum across architectures I think. But what about in the single-path case for Reptile/Metaprox?
        w_optimizer = torch.optim.SGD(network.weights, lr = orig_w_optimizer.param_groups[0]['lr'], momentum = 0.9, dampening = 0, weight_decay = 5e-4, nesterov = False)
      else:
        w_optimizer = torch.optim.SGD(network.weights, lr = orig_w_optimizer.param_groups[0]['lr'], momentum = 0.9, dampening = 0, weight_decay = 5e-4, nesterov = False)
      logger.log(f"Reinitalized w_optimizer for use in Reptile/Metaprox to make sure it does not have momentum etc. into {w_optimizer}")
  else:
    model_init, w_optim_init = None, None
  before_rollout_state = {} # Holds state inbetween outer loop rollouts so that we can come back to the initialization (eg. for bilevel optim)
  before_rollout_state["model_init"] = model_init
  before_rollout_state["w_optim_init"] = w_optim_init  
  arch_overview = {"cur_arch": None, "all_cur_archs": [], "all_archs": [], "top_archs_last_epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
  search_loader_iter = iter(xloader)
  if xargs.inner_steps is not None:
    inner_steps = xargs.inner_steps
  else:
    inner_steps = 1 # SPOS equivalent
  logger.log(f"Starting search with batch_size={len(next(iter(xloader))[0])}, len={len(xloader)}")
  
  use_higher_cond, diffopt_higher_grads_cond, monkeypatch_higher_grads_cond, \
  first_order_grad_for_free_cond, first_order_grad_concurrently_cond, second_order_grad_optimization_cond = resolve_higher_conds(xargs)
  
  logger.log(f"Use_higher_cond={use_higher_cond}, monkeypatch_higher_grads_cond={monkeypatch_higher_grads_cond}, diffopt_higher_grads_cond={diffopt_higher_grads_cond}")

  for data_step, (base_inputs, base_targets, arch_inputs, arch_targets) in tqdm(enumerate(search_loader_iter), desc = "Iterating over SearchDataset", total = round(len(xloader)/(inner_steps if not xargs.inner_steps_same_batch else 1))): # Accumulate gradients over backward for sandwich rule
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, xargs)
    network.zero_grad()
    if (smoke_test and data_step >= 3) or (xargs.steps_per_epoch_supernet is not None and data_step >= xargs.steps_per_epoch_supernet):
      break
    if data_step == 0:
      logger.log(f"New epoch (len={len(search_loader_iter)}) of arch; for debugging, those are the indexes of the first minibatch in epoch: {base_targets[0:10]}")
    scheduler.update(None, 1.0 * data_step / len(xloader))
    
    outer_iters = 1 if (xargs.sandwich is None or xargs.sandwich == 1) else xargs.sandwich
    inner_rollouts, meta_grads = [], [] # For implementing meta-batch_size in Reptile/MetaProx and similar
    if xargs.sandwich_mode in ["quartiles", "fairnas"]:
      if xargs.search_space_paper == "nats-bench":
        sampled_archs = arch_sampler.sample(mode = xargs.sandwich_mode, subset = all_archs, candidate_num=max(xargs.sandwich if xargs.sandwich is not None else 1, 
                                                                                                              xargs.inner_sandwich if xargs.inner_sandwich is not None else 1)) # Always samples 4 new archs but then we pick the one from the right quartile
      elif xargs.search_space_paper == "darts" and xargs.sandwich_mode == "fairnas":
        sampled_archs = network.sample_archs_fairnas()
    elif xargs.sandwich is not None and xargs.sandwich > 1 and 'gdas' not in xargs.algo:
      if arch_sampler is not None:
        sampled_archs = arch_sampler.sample(mode = "random", subset = all_archs, candidate_num=max(xargs.sandwich if xargs.sandwich is not None else 1, 
                                                                                                            xargs.inner_sandwich if xargs.inner_sandwich is not None else 1)) # Always samples 4 new archs but then we pick the one from the right quartile
      else:
        sampled_archs = [network.sample_arch() for _ in range(xargs.sandwich)]
    elif ((xargs.sandwich is not None and xargs.sandwich >= 1) or (xargs.inner_steps is not None and xargs.inner_steps > 1)) and 'gdas' in xargs.algo:
      sampled_archs = network.sample_gumbels(k=xargs.sandwich if xargs.sandwich is not None else 1)
    else:
      sampled_archs = None
      
    for outer_iter in range(outer_iters):
      # Update the weights
      if xargs.meta_algo in ['reptile', 'metaprox'] and outer_iters >= 1: # In other cases, we use Higher which does copying in each rollout already, so we never contaminate the initial network state
        if data_step <= 1:
          logger.log(f"""After restoring original params at outer_iter={outer_iter}, data_step={data_step}: Original net: {str(list(before_rollout_state['model_init'].parameters())[1])[0:80]}, {str(list(network.parameters())[5])[0:80]}, 
                  after rollout net: {str(list(network.parameters())[1])[0:80]}, {str(list(network.parameters())[5])[0:80]}""")
          logger.log(f"Arch check: Original net: {str(list(before_rollout_state['model_init'].alphas))[0:80]}, after-rollout net: {str(list(network.alphas))[0:80]}")

        if outer_iters > 1: # Dont need to reload the initial model state when meta batch size is 1
          network.load_state_dict(before_rollout_state["model_init"].state_dict())
        # w_optimizer.load_state_dict(before_rollout_state["w_optim_init"].state_dict())
      sampled_arch = sample_arch_and_set_mode_search(args=xargs, outer_iter=outer_iter, sampled_archs=sampled_archs, api=api, network=network, algo=algo, arch_sampler=arch_sampler, 
                                                     step=data_step, logger=logger, epoch=epoch, supernets_decomposition=supernets_decomposition, 
                                                     all_archs=all_archs, arch_groups_brackets=arch_groups_brackets)
      
      # TODO Put it in there even if None to make it act as a counter of sampled archs
      arch_overview["cur_arch"] = sampled_arch
      arch_overview["all_archs"].append(sampled_arch)
      arch_overview["all_cur_archs"].append(sampled_arch)

      weights_mask = [1 if ('arch' not in n and 'alpha' not in n) else 0 for (n, p) in network.named_parameters()] # Zeroes out all the architecture gradients in Higher. It has to be hacked around like this due to limitations of the library
      zero_arch_grads = lambda grads: [g*x if g is not None else 0 for g,x in zip(grads, weights_mask)]
      if use_higher_cond: 
        # NOTE first order algorithms have separate treatment because they are much sloer with Higher TODO if we want faster Reptile/Metaprox, should we avoid Higher? But makes more potential for mistakes
        fnetwork = higher.patch.monkeypatch(network, device='cuda', copy_initial_weights=True if xargs.higher_loop == "bilevel" else False, track_higher_grads = monkeypatch_higher_grads_cond)
        diffopt = higher.optim.get_diff_optim(w_optimizer, network.parameters(), fmodel=fnetwork, grad_callback=zero_arch_grads, device='cuda', override=None, track_higher_grads = diffopt_higher_grads_cond) 
        fnetwork.zero_grad() # TODO where to put this zero_grad? was there below in the sandwich_computation=serial branch, tbut that is surely wrong since it wouldnt support higher meta batch size
      else: 
        fnetwork = network
        diffopt = w_optimizer

      sotl, first_order_grad, second_order_grad_optimization, train_data_buffer, proximal_penalty  = [], None, None, [[], []], None
      assert inner_steps == 1 or xargs.meta_algo is not None or xargs.implicit_algo is not None
      assert xargs.meta_algo is None or (xargs.higher_loop is not None or xargs.meta_algo in ['reptile', 'metaprox'])
      assert xargs.sandwich is None or xargs.inner_sandwich is None # TODO implement better for meta-meta-batch sizes?
      assert all_archs is None or sampled_arch in all_archs 

      if xargs.meta_algo is not None and "higher" in xargs.meta_algo:
        assert xargs.higher_order is not None
      inner_sandwich_steps = xargs.inner_sandwich if xargs.inner_sandwich is not None else 1
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in tqdm(enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)), desc="Iterating over inner batches", 
                                                                                     disable=True if round(len(xloader)/(inner_steps if not xargs.inner_steps_same_batch else 1)) > 1 else False, total= len(all_base_inputs)):
        for inner_sandwich_step in range(inner_sandwich_steps):
          # NOTE for MultiPath, this also changes it to the appropriate n-th sampled arch, it does not sample new ones!
          # if inner_sandwich_steps > 1: # Was sampled above in the outer loop already. This might overwrite it in when using Inner Sandwich
          sampled_arch = sample_arch_and_set_mode_search(args=xargs, outer_iter=inner_sandwich_step, sampled_archs=sampled_archs, api=api, network=network, 
                                                        algo=algo, arch_sampler=arch_sampler, 
                                                    step=data_step, logger=logger, epoch=epoch, supernets_decomposition=supernets_decomposition, 
                                                    all_archs=all_archs, arch_groups_brackets=arch_groups_brackets
                                                    )
          if data_step < 2 and inner_step < 2 and epoch < 4 and outer_iter < 3:
            logger.log(f"Base targets in the inner loop at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}, arch_targets={arch_targets[0:10] if arch_targets is not None else None}")
            if inner_step == 1: logger.log(f"Arch during inner_steps: Original net: {str(list(network.alphas))[0:80]}")
            if xargs.inner_steps is not None and xargs.inner_steps > 1 and ('gdas' in xargs.algo or (xargs.meta_algo is not None and 'gdas' in xargs.meta_algo)) and hasattr(fnetwork, "last_gumbels"):
              logger.log(f"Supernet gumbels at data_step={data_step}, inner_step={inner_step}, epoch={epoch}, outer_iter={outer_iter}, refresh_arch={fnetwork.refresh_arch_oneshot}: {fnetwork.last_gumbels}")
          _, logits = fnetwork(base_inputs)
          base_loss = criterion(logits, base_targets) * (1 if xargs.sandwich is None else 1/xargs.sandwich)
          sotl.append(base_loss)

          if outer_iter == outer_iters - 1 and replay_buffer is not None and xargs.replay_buffer > 0: # We should only do the replay once regardless of the architecture batch size
            # TODO need to implement replay support for DARTS space (in general, for cases where we do not get an arch directly but instead use uniform sampling at each choice block)
            for replay_arch in replay_buffer:
              fnetwork.set_cal_mode('dynamic', replay_arch)
              _, logits = fnetwork(base_inputs)
              replay_loss = criterion(logits, base_targets)
              if epoch in [0,1] and data_step == 0:
                logger.log(f"Replay loss={replay_loss.item()} for {len(replay_buffer)} items with num_iters={outer_iters}, outer_iter={outer_iter}, replay_buffer={replay_buffer}") # Debugging messages
              base_loss = base_loss + (xargs.replay_buffer_weight / xargs.replay_buffer) * replay_loss # TODO should we also specifically add the L2 regularizations as separate items? Like this, it diminishes the importance of weight decay here
              fnetwork.set_cal_mode('dynamic', arch_overview["cur_arch"])
          if xargs.meta_algo == "metaprox":
            proximal_penalty = nn_dist(fnetwork, before_rollout_state["model_init"])
            if epoch % 5 == 0 and data_step in [0, 1]:
              logger.log(f"Proximal penalty at epoch={epoch}, step={data_step} was found to be {proximal_penalty} before applying lambda={xargs.metaprox_lambda}")
            base_loss = base_loss + xargs.metaprox_lambda/2*proximal_penalty # TODO scale by sandwich size?
          
          if xargs.meta_algo in ["reptile", "metaprox"] or xargs.implicit_algo:
            base_loss.backward()
            if inner_sandwich_steps == 1 or inner_sandwich_step == inner_sandwich_steps - 1:
              w_optimizer.step()
              w_optimizer.zero_grad()
              
          elif (not xargs.meta_algo):
            base_loss.backward() # Accumulate gradients over outer. There is supposed to be no training in inner loop!
          

          elif xargs.meta_algo and xargs.meta_algo not in ['reptile', 'metaprox']: # Gradients using Higher
            assert use_higher_cond
            new_params, cur_grads = diffopt.step(base_loss)
            cur_grads = list(cur_grads)
            for idx, (g, p) in enumerate(zip(cur_grads, fnetwork.parameters())):
              if g is None:
                cur_grads[idx] = torch.zeros_like(p)
          
            first_order_grad = fo_grad_if_possible(args=xargs, fnetwork=fnetwork, criterion=criterion, 
                                                  all_arch_inputs=all_arch_inputs, all_arch_targets=all_arch_targets, 
                                                  arch_inputs=arch_inputs, arch_targets=arch_targets, cur_grads=cur_grads,
                                                  inner_step=inner_step, inner_steps=inner_steps,
                                                  step=data_step, outer_iter=outer_iter,
                                                  first_order_grad=first_order_grad, first_order_grad_for_free_cond=first_order_grad_for_free_cond,
                                                  first_order_grad_concurrently_cond=first_order_grad_concurrently_cond, logger=logger)
            if second_order_grad_optimization_cond and inner_step == 0:
              second_order_grad_optimization = cur_grads
              if epoch < 2 and data_step < 3:
                logger.log(f"Using second_order_grad optimization by reusing the first grad from training (possible with higher_method=sotl) and not recomputing it; head={second_order_grad_optimization[0]}")
          else:
            pass # Standard multi-path branch. We do not update here because we want to accumulate grads over outer_iters before any updates

          if 'gradnorm' in algo: # Normalize gradnorm so that all updates have the same norm. But does not work well at all in practice
            coef, total_norm = grad_scale(w_optimizer.param_groups[0]["params"], grad_norm_meter.avg)
            grad_norm_meter.update(total_norm)

          if supernets_decomposition is not None:
            update_supernets_decomposition(supernets_decomposition=supernets_decomposition, arch_groups_quartiles=arch_groups_quartiles, losses_percs=losses_percs,
                                          sampled_arch=sampled_arch, fnetwork=fnetwork)
          base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
          if inner_step == 0 or xargs.implicit_algo is not None:
            base_losses.update(base_loss.item() / (1 if xargs.sandwich is None else 1/xargs.sandwich),  base_inputs.size(0))
            base_top1.update  (base_prec1.item(), base_inputs.size(0))
            base_top5.update  (base_prec5.item(), base_inputs.size(0))
            arch_overview["train_acc"].append(base_prec1)
            arch_overview["train_loss"].append(base_loss.item())
      
                
      meta_grads, inner_rollouts = hypergrad_outer(args=xargs, fnetwork=fnetwork, criterion=criterion, arch_targets=arch_targets, arch_inputs=arch_inputs,
                                                   all_arch_inputs=all_arch_inputs, all_arch_targets=all_arch_targets, all_base_inputs=all_base_inputs, all_base_targets=all_base_targets,
                                                   sotl=sotl, inner_step=inner_step, inner_steps=inner_steps, inner_rollouts=inner_rollouts, second_order_grad_optimization=second_order_grad_optimization,
                                                   first_order_grad_for_free_cond=first_order_grad_for_free_cond, first_order_grad_concurrently_cond=first_order_grad_concurrently_cond,
                                                   monkeypatch_higher_grads_cond=monkeypatch_higher_grads_cond, zero_arch_grads_lambda=zero_arch_grads, meta_grads=meta_grads,
                                                   step=data_step, epoch=epoch, logger=logger)
      if epoch < 2 and data_step < 3 and outer_iter < 3:
        logger.log(f"After running hypergrad_outer: len of meta_grads={len(meta_grads)}, len of inner_rollouts={len(inner_rollouts)}")
      if first_order_grad is not None:
        assert first_order_grad_for_free_cond or first_order_grad_concurrently_cond
        if epoch < 2 and data_step < 3:
          logger.log(f"""Putting first_order_grad into meta_grads (NOTE we aggregate first order grad by summing in the first place to save memory, so dividing by inner steps gives makes it average over the rollout) 
                     (len of first_order_grad ={len(first_order_grad)}, len of param list={len(list(network.parameters()))}),
                     with reduction={xargs.higher_reduction}, inner_steps (which is the division factor)={inner_steps}, outer_iters={outer_iters}, head={first_order_grad[0]}""")
        if xargs.higher_reduction == "sum": # the first_order_grad is computed in a way that equals summing
          meta_grads.append(first_order_grad)
        else:
          meta_grads.append([g/inner_steps if g is not None else g for g in first_order_grad])
      
      if brackets_cond:
        update_brackets(supernet_train_stats_by_arch, supernet_train_stats, supernet_train_stats_avgmeters, arch_groups_brackets, arch_overview, 
          [("train_loss", base_loss.item() / (1 if xargs.sandwich is None else 1/xargs.sandwich)), ("train_acc", base_prec1.item())], all_brackets, sampled_arch,  xargs)
        
    if xargs.meta_algo is None and xargs.algo != "darts-single":
      # The standard multi-path branch. Note we called base_loss.backward() earlier for this meta_algo-free code branch since meta_algo-free algos (SPOS, FairNAS, ..) do not do any training in inner steps
      
      with torch.no_grad():
        grad_drop(network.weights, p = xargs.grad_drop_p)
      w_optimizer.step()
      network.zero_grad()
      
    elif xargs.algo == "darts-single":
      a_optimizer.step()
      w_optimizer.step()
      w_optimizer.zero_grad()
      a_optimizer.zero_grad()
      network.zero_grad()

    # ARCHITECTURE/META-WEIGHTS UPDATE STEP. Updating archs after all weight updates in the unrolling are finished
    
    # for previously_sampled_arch in arch_overview["all_cur_archs"]:
    for previously_sampled_arch in [arch_overview["all_cur_archs"][-1]]: # TODO think about what to do with this. Delete completely?
      arch_loss = torch.tensor(10) # Placeholder in case it never gets updated here. It is not very useful in any case
      if algo.startswith("setn"):
        network.set_cal_mode('joint')
      elif algo.startswith('gdas'):
        network.set_cal_mode('gdas', None)
      elif algo.startswith('darts'):
        network.set_cal_mode('joint', None)
      elif 'random' in algo and len(arch_overview["all_cur_archs"]) > 1 and xargs.replay_buffer is not None:
        network.set_cal_mode('dynamic', previously_sampled_arch)
      elif 'random' in algo:
        network.set_cal_mode('urs', None)
      elif algo != 'enas':
        raise ValueError('Invalid algo name : {:}'.format(algo))
      network.zero_grad()
      if algo == 'darts-v2' and not xargs.meta_algo:
        start = time.time()
        if xargs.search_space_paper == "nats-bench":
          arch_loss, logits = backward_step_unrolled(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets)
        elif xargs.search_space_paper == "darts":
          arch_loss, logits = backward_step_unrolled_darts(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets)
        # print(f"TIME OF BACKWARD UNROLLED: {time.time()-start}")
        a_optimizer.step()
      elif (algo == 'random' or algo == 'enas' or 'random' in algo ) and not xargs.meta_algo and not xargs.implicit_algo:
        if algo == "random" and xargs.merge_train_val_supernet:
          arch_loss = torch.tensor(10) # Makes it slower and does not return anything useful anyways
        else:
          with torch.no_grad():
            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
      elif xargs.meta_algo:
        avg_meta_grad = hyper_meta_step(network, inner_rollouts, meta_grads, xargs, data_step, 
                                        logger, before_rollout_state["model_init"], outer_iters, outer_iter, epoch)
        
        network.load_state_dict(
          before_rollout_state["model_init"].state_dict())  # Need to restore to the pre-rollout state before applying meta-update
        # The architecture update itself
        with torch.no_grad():  # Update the pre-rollout weights
            for (n, p), g in zip(network.named_parameters(), avg_meta_grad):
                cond = ('arch' not in n and 'alpha' not in n) if xargs.higher_params == "weights" else ('arch' in n or 'alpha' in n)  # The meta grads typically contain all gradient params because they arise as a result of torch.autograd.grad(..., model.parameters()) in Higher
                if cond:
                    if g is not None and p.requires_grad:
                      if type(g) is int and g == 0:
                        p.grad = torch.zeros_like(p)
                      else:
                        p.grad = g

                else:
                  p.grad = None
        meta_optimizer.step()
        meta_optimizer.zero_grad()
        pass
      elif xargs.implicit_algo:
        # NOTE hyper_step also stores the grads into arch_params_real directly
        hyper_loss, hyper_grads = implicit_step(model=fnetwork, train_loader=train_loader, val_loader=val_loader, criterion=criterion, 
                                             arch_params=fnetwork.alphas, arch_params_real=network.alphas,
                                             elementary_lr=w_optimizer.param_groups[0]['lr'], max_iter=xargs.implicit_steps, algo=xargs.implicit_algo)
        print(f"Hyper grads: {hyper_grads}")
        if xargs.implicit_grad_clip is not None:
          clip_coef = torch.nn.utils.clip_grad_norm_(network.alphas, xargs.implicit_grad_clip, norm_type=2.0)
          print(f"Clipped implicit grads by {clip_coef}")
        a_optimizer.step()
      elif xargs.algo == "darts-single":
        pass
      else:
        # The Darts-V1/FOMAML/GDAS/who knows what else branch
        network.zero_grad()
        _, logits = network(arch_inputs)
        arch_loss = criterion(logits, arch_targets)
        arch_loss.backward()
        a_optimizer.step()

      #### TRAINING WEIGHTS WITH UPDATED ARCHITECTURE (as the final step of bilevel optimization)

      if xargs.higher_method in ["val_multiple_v2", "sotl_v2"]: # Fake SoVL without stochasticity by using the architecture training data for weights training in the real-weight-training step
        all_base_inputs, all_base_targets = all_arch_inputs, all_arch_targets
      # Train the weights for real if necessary (in bilevel loops, say). NOTE this skips Reptile/metaprox because they have higher_params=weights
      if use_higher_cond and xargs.higher_loop == "bilevel" and xargs.higher_params == "arch" and xargs.sandwich_computation == "serial" and xargs.meta_algo not in ["reptile", "metaprox"]:
        if xargs.bilevel_refresh_arch: network.refresh_arch_oneshot = True
        for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
          if inner_step == 1 and xargs.inner_steps_same_batch: # TODO Dont need more than one step of finetuning when using a single batch for the bilevel rollout I think?
            break
          if xargs.bilevel_train_steps is not None and inner_step >= xargs.bilevel_train_steps:
            break
          if data_step in [0, 1] and inner_step < 3 and epoch < 5:
            logger.log(f"Doing weight training for real in higher_loop={xargs.higher_loop} at inner_step={inner_step}, step={data_step}: target={base_targets[0:10]}")
            logger.log(f"Weight-training-for-real check: Original net: {str(list(before_rollout_state['model_init'].parameters())[1])[0:80]}, after-rollout net: {str(list(network.parameters())[1])[0:80]}")
            logger.log(f"Arch check: Original net: {str(list(before_rollout_state['model_init'].alphas))[0:80]}, after-rollout net: {str(list(network.alphas))[0:80]}")
          _, logits = network(base_inputs)
          base_loss = criterion(logits, base_targets) * (1 if xargs.sandwich is None else 1/xargs.sandwich)
          network.zero_grad()
          base_loss.backward()
          w_optimizer.step()
        if xargs.bilevel_refresh_arch: network.refresh_arch_oneshot = False

      elif use_higher_cond and xargs.higher_loop == "joint" and xargs.higher_loop_joint_steps is None and xargs.higher_params == "arch" and xargs.sandwich_computation == "serial" and outer_iter == outer_iters - 1 and xargs.meta_algo not in ["reptile", "metaprox"]:
        if epoch == 0 and data_step < 3:
          logger.log(f"Updating meta-weights by copying from the rollout model")
        with torch.no_grad():
          for (n1, p1), p2 in zip(network.named_parameters(), fnetwork.parameters()):
            if ('arch' not in n1 and 'alpha' not in n1): # Want to copy weights only - the architecture update was done on the original network
              p1.data = p2.data
      elif use_higher_cond and xargs.higher_loop == "joint" and xargs.higher_loop_joint_steps is not None and xargs.higher_params == "arch" and xargs.sandwich_computation == "serial" and outer_iter == outer_iters - 1 and xargs.meta_algo not in ["reptile", "metaprox"]:
        # This branch can be used for GDAS with unrolled SOTL
        for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
          if inner_step >= xargs.higher_loop_joint_steps:
            break
          if data_step < 2 and inner_step < 3 and epoch < 5:
            logger.log(f"Doing weight training for real in higher_loop={xargs.higher_loop} with higher_loop_joint_steps={xargs.higher_loop_joint_steps} at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
            logger.log(f"Arch check: Original net: {str(list(before_rollout_state['model_init'].alphas))[0:80]}, after-rollout net: {str(list(network.alphas))[0:80]}")
          _, logits = network(base_inputs)
          base_loss = criterion(logits, base_targets) * (1 if xargs.sandwich is None else 1/xargs.sandwich)
          network.zero_grad()
          base_loss.backward()
          w_optimizer.step()

      if xargs.meta_algo and use_higher_cond:
        del fnetwork
        del diffopt

      # record
      if arch_targets is not None:
        arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
        val_batch_size = arch_inputs.size(0)
      else:
        arch_prec1, arch_prec5, val_batch_size = torch.tensor(0), torch.tensor(0), torch.tensor(1)
      arch_losses.update(arch_loss.item(),  val_batch_size)
      arch_top1.update  (arch_prec1.item(), val_batch_size)
      arch_top5.update  (arch_prec5.item(), val_batch_size)
      arch_overview["val_acc"].append(arch_prec1)
      arch_overview["val_loss"].append(arch_loss.item())

      if brackets_cond:
        update_brackets(supernet_train_stats_by_arch, supernet_train_stats, supernet_train_stats_avgmeters, arch_groups_brackets, arch_overview, 
          [("val_loss", arch_loss.item()), ("val_acc", arch_prec1.item())], all_brackets, sampled_arch,  xargs)

    if xargs.meta_algo is not None: # NOTE this is the end of outer loop; will start new episode soon. We update the pre-rollout state for next iteration
      if data_step <= 1:
        logger.log(f"Before reassigning model_init at outer_iter={outer_iter}, data_step={data_step}: Original net: {str(list(before_rollout_state['model_init'].parameters())[1])[0:80]}, after-rollout net: {str(list(network.parameters())[1])[0:80]}")
        logger.log(f"Arch check: Original net: {str(list(before_rollout_state['model_init'].alphas))[0:80]}, after-rollout net: {str(list(network.alphas))[0:80]}")

      # model_init = deepcopy(network) # Need to make another copy of initial state for rollout-based algorithms
      # w_optim_init = deepcopy(w_optimizer)
      # before_rollout_state["model_init"] = model_init
      before_rollout_state["model_init"].load_state_dict(network.state_dict())
      before_rollout_state["w_optim_init"] = w_optim_init

    arch_overview["all_cur_archs"] = [] #Cleanup
    network.zero_grad()
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if data_step % print_freq == 0 or data_step + 1 == len(xloader) or xargs.implicit_algo is not None:
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, data_step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  
  if xargs.hessian and algo.startswith('darts') and torch.cuda.get_device_properties(0).total_memory > (20147483648 if xargs.max_nodes < 7 else 9147483648) and xargs.search_space_paper != "darts": # Crashes with just 8GB of memory
    eigenvalues = exact_hessian(network, val_loader, criterion, xloader, epoch, logger, xargs)
  elif xargs.hessian and algo.startswith('darts') and ((torch.cuda.get_device_properties(0).total_memory < 9147483648 and xargs.search_space_paper != "darts") or (torch.cuda.get_device_properties(0).total_memory > 9147483648 and xargs.search_space_paper == "darts")):
    eigenvalues = approx_hessian(network, val_loader, criterion, xloader, xargs)
  else:
    eigenvalues = None

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
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg, supernet_train_stats, supernet_train_stats_by_arch, arch_overview, search_metric_stds, eigenvalues



def get_best_arch(train_loader, valid_loader, network, n_samples, algo, logger, criterion,
  additional_training=True, api=None, style:str='val_acc', w_optimizer=None, w_scheduler=None, 
  config: Dict=None, epochs:int=1, steps_per_epoch:int=100, 
  val_loss_freq:int=2, train_stats_freq=3, overwrite_additional_training:bool=False, 
  scheduler_type:str=None, xargs:Namespace=None, train_loader_stats=None, val_loader_stats=None, 
  model_config=None, all_archs=None, search_sotl_stats=None, checkpoint_freq=1, search_epoch=None):
  
  true_archs = None
  with torch.no_grad():
    network.eval()
    if 'random' in algo:
      if xargs.evenly_split is not None:
        arch_sampler = ArchSampler(api=api, model=network, mode=xargs.evenly_split, dataset = xargs.evenly_split_dset, op_names = network._op_names, max_nodes = xargs.max_nodes, search_space = xargs.search_space_paper)
        archs = arch_sampler.sample(mode="evenly_split", candidate_num=xargs.eval_candidate_num)
        decision_metrics = []
      elif api is not None and xargs is not None:
        archs, decision_metrics = network.return_topK(n_samples, True, api=api, dataset=xargs.dataset, size_percentile=xargs.size_percentile, perf_percentile=xargs.perf_percentile), []
      else:
        archs, decision_metrics = network.return_topK(n_samples, True), []
      if xargs.archs_split is not None:
        logger.log(f"Loading archs from {xargs.archs_split} to use as sampled architectures in finetuning with algo={algo}")
        with open(f'./configs/nas-benchmark/arch_splits/{xargs.archs_split}', 'rb') as f:
          archs = pickle.load(f)
      elif xargs.save_archs_split is not None:
        logger.log(f"Savings archs split to {xargs.archs_split} to use as sampled architectures in finetuning with algo={algo}")
        if not os.path.exists(f'./configs/nas-benchmark/arch_splits/{xargs.save_archs_split}'):
          with open(f'./configs/nas-benchmark/arch_splits/{xargs.save_archs_split}', 'wb') as f:
            pickle.dump(archs, f)
        else:
          logger.log(f"The path to be saved to at './configs/nas-benchmark/arch_splits/{xargs.save_archs_split}' already exists! Not saving archs_split")
    elif algo.startswith("setn"):
      logger.log(f"Sampled {n_samples} SETN architectures using the Template network")
      archs, decision_metrics = network.return_topK(n_samples, False), []
    elif algo.startswith('darts'):
      arch = network.get_genotype(original_darts_format=True)
      true_archs, true_decision_metrics = [arch], [] # Put the same arch there twice for the rest of the code to work in idempotent way
      archs, decision_metrics = network.return_topK(n_samples, False, api=api, dataset=xargs.dataset, size_percentile=xargs.size_percentile, perf_percentile=xargs.perf_percentile), []

    elif algo.startswith("gdas"):
      # Remember - GDAS is argmax on forward, softmax on backward. Putting random=False in return_topK makes it return archs ordered by log probability, which starts with the argmax arch and then the next most probable
      arch = network.get_genotype(original_darts_format=True)
      true_archs, true_decision_metrics = [arch], [] # Put the same arch there twice for the rest of the code to work in idempotent way
      archs, decision_metrics = network.return_topK(n_samples, False, api=api, dataset=xargs.dataset, size_percentile=xargs.size_percentile, perf_percentile=xargs.perf_percentile), []
    elif algo == 'enas':
      archs, decision_metrics = [], []
      for _ in range(n_samples):
        _, _, sampled_arch = network.controller()
        archs.append(sampled_arch)
    else:
      raise ValueError('Invalid algorithm name : {:}'.format(algo))
    
    if xargs.search_space_paper in ["nats-bench"]:
      if all_archs is not None and not xargs.archs_split: # Overwrite the just sampled archs with the ones that were supplied. Useful in order to match up with the archs used in search_func
        logger.log(f"Overwrote arch sampling in get_best_arch with a subset of len={len(all_archs)}, head = {[api.archstr2index[arch.tostr()] for arch in all_archs[0:10]]}")
        archs = all_archs
      else:
        logger.log(f"Were not supplied any limiting subset of archs so instead just sampled fresh ones with len={len(archs)}, head = {[api.archstr2index[arch.tostr()] for arch in archs[0:10]]} using algo={algo}")
      logger.log(f"Running get_best_arch (evenly_split={xargs.evenly_split}, style={style}, evenly_split_dset={xargs.evenly_split_dset}) with initial seeding of archs head:{[api.archstr2index[arch.tostr()] for arch in archs[0:10]]}")
      
    # TODO dont need this? Makes it very slow for NB301 as well
    # # The true rankings are used to calculate correlations later
    # if style in ["sotl"]:
    #   true_rankings, final_accs = get_true_rankings(archs, api)
    #   # true_rankings_rounded, final_accs_rounded = get_true_rankings(archs, api, decimals=3) # np.round(0.8726, 3) gives 0.873, ie. we wound accuracies to nearest 0.1% 
    # elif style in ["val", "val_acc"]:
    #   true_rankings, final_accs = get_true_rankings(archs, api, is_random=True)
    # else:
    #   raise NotImplementedError

    if true_archs is not None:
      true_rankings_final, final_accs_final = get_true_rankings(true_archs, api)
      assert len(true_archs) == 1
      wandb.log({"final":final_accs_final[true_archs[0].tostr()], "epoch": search_epoch}) # Log the final selected arch accuracy by GDAS/DARTS as separate log entry
    
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
          # if metric == "kl" and not ('darts' in xargs.algo): # Does not make sense to use KL divergence if we do not train the whole supernet - TODO maybe it does though?
          #   continue
          cur_loader = valid_loader if data_type == "val" else train_loader

          decision_metrics_computed, decision_sum_metrics_computed = eval_archs_on_batch(xloader=cur_loader, archs=archs[0:200], network=network, criterion=criterion, metric=metric, 
            train_loader=train_loader, w_optimizer=w_optimizer, train_steps=xargs.eval_arch_train_steps, same_batch = True) 

          best_idx_search = np.argmax(decision_metrics_computed)
          best_arch_search, best_valid_acc_search = archs[best_idx_search], decision_metrics_computed[best_idx_search]
          search_results_top1 = summarize_results_by_dataset(best_arch_search, api=api, iepoch=199, hp='200')
          
          decision_metrics_eval["supernet_" + data_type + "_" + metric] = decision_metrics_computed

          search_summary_stats["search"][data_type][metric]["mean"] = np.mean(decision_metrics_computed)
          search_summary_stats["search"][data_type][metric]["std"] = np.std(decision_metrics_computed)
          search_summary_stats["search"][data_type][metric]["top1"] = search_results_top1

      try:
        decision_metrics = decision_metrics_eval["supernet_val_acc"]
      except Exception as e:
        logger.log(f"Failed to get decision metrics - decision_metrics_eval={decision_metrics_eval}")
      wandb.log({**corrs, **search_summary_stats})
    else:
      decision_metrics, decision_sum_metrics = eval_archs_on_batch(xloader=valid_loader, archs=archs, network=network, 
                                                                  train_loader=train_loader, w_optimizer=w_optimizer, train_steps = xargs.eval_arch_train_steps)

  if style == 'sotl' or style == "sovl":
    true_rankings, final_accs = get_true_rankings(archs, api)
    upper_bound = {}
    for n in [1,5,10]:
      upper_bound[f"top{n}"] = {"cifar10":0, "cifar10-valid":0, "cifar100":0, "ImageNet16-120":0}
      for dataset in true_rankings.keys():
        upper_bound[f"top{n}"][dataset] += sum([x["metric"] for x in true_rankings[dataset][0:n]])/min(n, len(true_rankings[dataset][0:n]))
    upper_bound = {"upper":upper_bound}
    logger.log(f"Upper bound: {upper_bound}")
    # true_rankings_rounded, final_accs_rounded = get_true_rankings(archs, api, decimals=3) # np.round(0.8726, 3) gives 0.873, ie. we wound accuracies to nearest 0.1% 
    # Branch for the single-architecture finetuning in order to collect SoTL    
    if xargs.postnet_switch_train_val:
      logger.log("Switching train and validation sets for postnet training. Useful for training on the test set if desired")
      train_loader, valid_loader = valid_loader, train_loader
    # Simulate short training rollout to compute SoTL for candidate architectures
    cond = logger.path('corr_metrics').exists() and not overwrite_additional_training
    total_metrics_keys = ["total_val", "total_train", "total_val_loss", "total_train_loss", "total_arch_count"]
    so_metrics_keys = ["sotl", "sovl", "sovalacc", "sotrainacc", "sovalacc_top5", "sogn", "sogn_norm"]
    grad_metric_keys = ["gn", "grad_normalized", "grad_mean_accum", "grad_accum"] # Deleted "grad_mean_sign"
    # pct_metric_keys = ["train_loss_pct"]
    pct_metric_keys = []
    metrics_keys = ["val_acc", "train_acc", "train_loss", "val_loss", "gap_loss", *pct_metric_keys, *grad_metric_keys, *so_metrics_keys, *total_metrics_keys]
    must_restart, config_no_restart_cond = False, True
    start_arch_idx = 0

    if cond: # Try to load previous checkpoint. It will restart if significant changes are detected in the current run from the checkpoint 
             # (this prevents accidentally using checkpoints for different params than the current ones)
      logger.log("=> loading checkpoint of the last-checkpoint '{:}' start".format(logger.path('corr_metrics')))
      try:
        checkpoint = torch.load(logger.path('corr_metrics'))
        logger.log(f"Loaded corr metrics checkpoint at {logger.path('corr_metrics')}")
      except Exception as e:
        logger.log("Failed to load corr_metrics checkpoint, trying backup now")
        try:
          checkpoint = torch.load(os.fspath(logger.path('corr_metrics'))+"_backup")
        except:
          must_restart=True
          pass

      try:
        metrics = {k:checkpoint["metrics"][k] for k in checkpoint["metrics"].keys()}
        train_stats = checkpoint["train_stats"]
      except Exception as e:
        print("Errored due to exception below")
        print(e)
        print("Unknown reason but must restart!")
        must_restart = True

      checkpoint_config = checkpoint["config"] if "config" in checkpoint.keys() else {}
      decision_metrics = checkpoint["decision_metrics"] if "decision_metrics" in checkpoint.keys() else []
      start_arch_idx = checkpoint["start_arch_idx"]
      cond1={k:v for k,v in checkpoint_config.items() if ('path' not in k and 'dir' not in k and k not in ["dry_run", "workers", "mmap", "search_logs_freq", "total_estimator_steps"])}
      cond2={k:v for k,v in vars(xargs).items() if ('path' not in k and 'dir' not in k and k not in ["dry_run", "workers", "mmap", "search_logs_freq", "total_estimator_steps"])}
      logger.log(f"Checkpoint config: {cond1}")
      logger.log(f"Newly input config: {cond2}")
      different_items = {k: cond1[k] for k in cond1 if k in cond2 and cond1[k] != cond2[k]}
      config_no_restart_cond = (cond1 == cond2 or len(different_items) == 0)
      if config_no_restart_cond:
        logger.log("Both configs are equal.")
      else:
        logger.log("Checkpoint and current config are not the same! need to restart")
        logger.log(f"Different items are : {different_items}")
      
      if set([x.tostr() if type(x) is not str else x for x in checkpoint["archs"]]) != set([x.tostr() if type(x) is not str else x for x in archs]):
        print("Checkpoint has sampled different archs than the current seed! Need to restart")
        print(f"Checkpoint: {checkpoint['archs'][0]}")
        print(f"Current archs: {archs[0]}")
        if all_archs is not None or xargs.archs_split is not None:
          logger.log(f"Architectures do not match up to the checkpoint but since all_archs (or archs_split={xargs.archs_split}) was supplied, it might be intended")
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
      logger.log(f"Evenifying the training so that all architectures have the equivalent of {max_epoch_attained} of training measured by their own training curves")

    if xargs.adaptive_lr:
      lr_counts = defaultdict(int)
    
    # TODO Check that everything is wroking correctly
    # network_init = deepcopy(network.state_dict()) # TODO seems unnecessary?
    logger.log(f"Starting finetuning at {start_arch_idx} with total len(archs)={len(archs)}")
    avg_arch_time = AverageMeter()
    for arch_idx, sampled_arch in tqdm(enumerate(archs[start_arch_idx:], start_arch_idx), desc="Iterating over sampled architectures", total = len(archs)-start_arch_idx):
      assert (all_archs is None) or (sampled_arch in all_archs), "There must be a bug since we are training an architecture that is not in the supplied subset"
      arch_start = time.time()
      arch_natsbench_idx = api.query_index_by_arch(sampled_arch)
      true_perf, true_step, arch_str = summarize_results_by_dataset(sampled_arch, api, separate_mean_std=False), 0, sampled_arch.tostr()
      arch_threshold = arch_rankings_thresholds_nominal[arch_rankings_thresholds[bisect.bisect_left(arch_rankings_thresholds, arch_rankings_dict[sampled_arch.tostr()]["rank"])]]
      if xargs.resample not in [False, None, "False", "false", "None"]:
        assert xargs.reinitialize
        search_model = get_cell_based_tiny_net(model_config, xargs)
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
        # network2 = network
        # network2.set_cal_mode('dynamic', sampled_arch)
        # network2.load_state_dict(network_init)
        if arch_idx < 3:
          try:
            logger.log(f"Finetuning-network sample weights {next((x for i, x in enumerate(network2.parameters()) if i == 2), None)}")
          except:
            logger.log("Logging finetuning-network sample weights failed; this is probably the fist iteration and the network has not been defined yet")
        network2 = deepcopy(network)
        if arch_idx < 3:
         logger.log(f"Deepcopied network with sample weights {next((x for i, x in enumerate(network.parameters()) if i == 2), None)}")
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
        best_lr = find_best_lr(xargs, network2, train_loader, config, arch_idx)
      else:
        best_lr = None
      if arch_idx < 3:
        logger.log(f"Picking the scheduler with scheduler_type={scheduler_type}, xargs.lr={xargs.lr}, xargs.postnet_decay={xargs.postnet_decay}")

      w_optimizer2, w_scheduler2, criterion = get_finetune_scheduler(scheduler_type, config, xargs, network2, best_lr, logger=logger)
      
      if arch_idx == start_arch_idx: #Should only print it once at the start of training
        logger.log(f"Optimizers for the supernet post-training: {w_optimizer2}, {w_scheduler2}")

      running = defaultdict(int)

      grad_metrics = init_grad_metrics(keys = ["train", "val", "total_train", "total_val"])

      start = time.time()
      train_loss_total, train_acc_total, _ = valid_func(xloader=train_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger,
                                                        steps=xargs.total_estimator_steps if not xargs.drop_fancy else 4, grads=xargs.grads_analysis)
      if xargs.grads_analysis and not xargs.drop_fancy:
        analyze_grads(network=network2, grad_metrics=grad_metrics["total_train"], true_step=true_step, arch_param_count=arch_param_count, 
                      zero_grads=True, total_steps=true_step)
      if not xargs.merge_train_val_postnet or (xargs.val_dset_ratio is not None and xargs.val_dset_ratio < 1):
        val_loss_total, val_acc_total, _ = valid_func(xloader=val_loader_stats, network=network2, criterion=criterion, algo=algo, 
                                                      logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
        if xargs.grads_analysis and not xargs.drop_fancy:
          analyze_grads(network=network2, grad_metrics=grad_metrics["total_val"], true_step=true_step, arch_param_count=arch_param_count, 
                        zero_grads=True, total_steps=true_step)
      else:
        val_loss_total, val_acc_total = train_loss_total, train_acc_total
      val_loss_total, train_loss_total = -val_loss_total, -train_loss_total
      logger.log(f"Computed total_metrics without grad metrics with {xargs.total_estimator_steps} steps, batch size={train_loader_stats.batch_size} in {time.time()-start} time")

      if not xargs.drop_fancy:
        grad_mean, grad_std = estimate_grad_moments(xloader=train_loader, network=network2, criterion=criterion, steps=5)
        grad_std_scalar = torch.mean(torch.cat([g.view(-1) for g in grad_std], dim=0)).item()
        grad_snr_scalar = (grad_std_scalar**2)/torch.mean(torch.pow(torch.cat([g.view(-1) for g in grad_mean], dim=0), 2)).item()
      else:
        grad_mean, grad_std, grad_std_scalar, grad_snr_scalar = 0, 0, 0, 0
      network2.zero_grad()

      if arch_idx < 5: # Dont need to print this for every architecture I guess
        logger.log(f"Time taken to compute total_train/total_val statistics once with {xargs.total_estimator_steps} estimator steps, batch size={train_loader_stats.batch_size} is {time.time()-start}")

      if xargs.individual_logs: # Log the training stats for each sampled architecture separately
        q = mp.Queue()
        # This reporting process is necessary due to WANDB technical difficulties. It is used to continuously report train stats from a separate process
        # Otherwise, when a Run is intiated from a Sweep, it is not necessary to log the results to separate training runs. But that it is what we want for the individual arch stats
        p=mp.Process(target=train_stats_reporter, kwargs=dict(queue=q, config=vars(xargs),
            sweep_group=f"Search_Cell_{algo}_arch", sweep_run_name=wandb.run.name or wandb.run.id or "unknown", sweep_id = wandb.run.sweep_id or "unknown", arch=sampled_arch.tostr()))
        p.start()

      if xargs.evenify_training:
        evenify_training(network2, train_loader, criterion, w_optimizer2, logger, arch_idx, epoch_eqs, sampled_arch)
    
      for epoch_idx in range(epochs):
        if epoch_idx < 5:
          logger.log(f"New epoch (len={len(train_loader)} and steps_per_epoch={steps_per_epoch}) of arch; for debugging, those are the indexes of the first minibatch in epoch with idx up to 5: {epoch_idx}: {next(iter(train_loader))[1][0:15]}")
          logger.log(f"Weights LR before scheduler update: {w_scheduler2.get_lr()[0]}")

        val_acc_evaluator = ValidAccEvaluator(valid_loader, None)
        total_metrics_dict = {"total_val":val_acc_total, "total_train":train_acc_total, "total_val_loss":val_loss_total, "total_train_loss": train_loss_total, "total_arch_count":arch_param_count, 
                        "total_gstd":grad_std_scalar, "total_gsnr":grad_snr_scalar}
        for batch_idx, data in tqdm(enumerate(train_loader), desc = "Iterating over batches", total=len(train_loader), disable=True):
          stop_early_cond = ((steps_per_epoch is not None and steps_per_epoch != "None") and batch_idx > steps_per_epoch)
          if stop_early_cond:
            break
          for metric, metric_val in total_metrics_dict.items():
            metrics[metric][arch_str][epoch_idx].append(metric_val)
        
          with torch.set_grad_enabled(mode=additional_training): # TODO FIX STEPS PER EPOCH SCHEUDLING FOR steps_per_epoch
            scheduler_step(w_scheduler2=w_scheduler2, epoch_idx=epoch_idx, batch_idx=batch_idx, 
                           train_loader=train_loader, steps_per_epoch=steps_per_epoch, scheduler_type=scheduler_type)

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
            
            if arch_param_count == -1: # Invalid placeholder value:
              arch_param_count = sum(p.numel() for p in network2.parameters() if p.requires_grad and p.grad is not None)
              logger.log(f"Estimated new param count as {arch_param_count}")

          true_step += 1

          if (batch_idx % val_loss_freq == 0) and (batch_idx % 100 == 0 or not xargs.drop_fancy):
            if (not xargs.merge_train_val_postnet) or xargs.postnet_switch_train_val or (xargs.val_dset_ratio is not None and xargs.val_dset_ratio < 1):
              w_optimizer2.zero_grad() # NOTE We MUST zero gradients both before and after doing the fake val gradient calculations
              valid_acc, valid_acc_top5, valid_loss = val_acc_evaluator.evaluate(arch=sampled_arch, network=network2, criterion=criterion, grads=xargs.grads_analysis)
              if xargs.grads_analysis:
                analyze_grads(network=network2, grad_metrics=grad_metrics["val"], true_step=true_step, arch_param_count=arch_param_count, total_steps=true_step)
              w_optimizer2.zero_grad() # NOTE We MUST zero gradients both before and after doing the fake val gradient calculations
            else:
              valid_acc, valid_acc_top5, valid_loss = 0, 0, 0

          running = update_running(running=running, valid_loss=valid_loss, valid_acc=valid_acc, valid_acc_top5=valid_acc_top5, loss=loss, 
                         train_acc_top1=train_acc_top1, train_acc_top5=train_acc_top5, sogn=grad_metrics["train"]["sogn"], 
                         sogn_norm=grad_metrics["train"]["grad_normalized"], total_train_loss_for_sotl_aug=total_metrics_dict["total_train_loss"])
          
          metrics = update_base_metrics(metrics=metrics, running=running, grad_metrics=grad_metrics, drop_fancy=xargs.drop_fancy, 
                              grads_analysis=xargs.grads_analysis, valid_acc=valid_acc, train_acc=train_acc_top1, loss=loss, 
                              valid_loss=valid_loss, arch_str=arch_str, epoch_idx=epoch_idx)
  

          if xargs.grads_analysis and not xargs.drop_fancy:
            metrics["gap_grad_accum"][arch_str][epoch_idx].append(metrics["train_grad_accum"][arch_str][epoch_idx][-1]-metrics["val_grad_accum"][arch_str][epoch_idx][-1])

          special_metrics = {k:metrics[k][arch_str][epoch_idx][-1] for k in metrics.keys() if len(metrics[k][arch_str][epoch_idx])>0}
          special_metrics = {**special_metrics, **{k+str(arch_threshold):v for k,v in special_metrics.items()}}
          batch_train_stats = {"lr":w_scheduler2.get_lr()[0], f"lr{arch_threshold}":w_scheduler2.get_lr()[0],
           "true_step":true_step, "train_loss":loss, f"train_loss{arch_threshold}":loss, 
          f"epoch_eq{arch_threshold}": closest_epoch(api, arch_str, loss, metric="train-loss")["epoch"] if xargs.search_space_paper in ['nats-bench'] else -1,
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

          # Refresh total_metrics once per some time because each evaluation takes ~20s
          if additional_training and (batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1) and batch_idx < 400 and not (batch_idx == 0 and epoch_idx == 0): # The initial values were already computed
            start = time.time()
            if not xargs.drop_fancy or xargs.merge_train_val_postnet:
              train_loss_total, train_acc_total, _ = valid_func(xloader=train_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps if not xargs.drop_fancy else 4, grads=xargs.grads_analysis)
              if xargs.grads_analysis:
                analyze_grads(network=network2, grad_metrics=grad_metrics["total_train"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)  
            network2.zero_grad() 
            if not xargs.merge_train_val_postnet or (xargs.val_dset_ratio is not None and xargs.val_dset_ratio < 1):
              val_loss_total, val_acc_total, _ = valid_func(xloader=val_loader_stats, network=network2, criterion=criterion, algo=algo, logger=logger, steps=xargs.total_estimator_steps, grads=xargs.grads_analysis)
            else:
              val_loss_total, val_acc_total = train_loss_total, train_acc_total
              if xargs.grads_analysis:
                analyze_grads(network=network2, grad_metrics=grad_metrics["total_val"], true_step=true_step, arch_param_count=arch_param_count, zero_grads=True, total_steps=true_step)
            val_loss_total, train_loss_total = -val_loss_total, -train_loss_total
            network2.zero_grad()
            grad_mean, grad_std = estimate_grad_moments(xloader=train_loader, network=network2, criterion=criterion, steps=5)
            grad_std_scalar = torch.mean(torch.cat([g.view(-1) for g in grad_std], dim=0)).item()
            grad_snr_scalar = (grad_std_scalar**2)/torch.mean(torch.pow(torch.cat([g.view(-1) for g in grad_mean], dim=0), 2)).item()
            network2.zero_grad()
            
            total_metrics_dict["total_val"], total_metrics_dict["total_train"] = val_acc_total, train_acc_total
            total_metrics_dict["total_val_loss"], total_metrics_dict["total_train_loss"] = val_loss_total, train_loss_total
            total_metrics_dict["total_gstd"], total_metrics_dict["total_gsnr"] = grad_std_scalar, grad_snr_scalar 

        #Cleanup at end of epoch
        grad_metrics["train"]["grad_accum_singleE"] = None
        grad_metrics["val"]["grad_accum_singleE"] = None
        if hasattr(train_loader.sampler, "reset_counter"): # Resetting counter is necessary for consistent epoch batch orders across architectures using the custom Sample. Also important for CIFAR5m
          train_loader.sampler.counter += 1

      final_metric = None # Those final/decision metrics are not very useful apart from being a compatibility layer with how get_best_arch worked in the base repo
      if style == "sotl":
        final_metric = running["sotl"]
      elif style == "sovl":
        final_metric = running["sovl"]

      decision_metrics.append(final_metric)
      
      if arch_idx % checkpoint_freq == 0 or arch_idx == len(archs)-1:
        try:
          corr_metrics_path = save_checkpoint({"corrs":{}, "metrics":metrics, "train_stats":train_stats,
            "archs":archs, "start_arch_idx": arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},   
            logger.path('corr_metrics'), logger, quiet=True, backup=False)
        except:
          print("Failed to save corr_metrics checkpoint! Will proceed but need ot be careful")

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

    if not xargs.drop_fancy:
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
      print(list(metrics.keys()))
      # print(metrics["total_val"])
      
      metrics_E1 = {metric+"E1": {arch.tostr():SumOfWhatever(measurements=metrics[metric][arch.tostr()], e=1).get_time_series(chunked=True, name=metric) for arch in archs} for metric,v in tqdm(metrics.items(), desc = "Calculating E1 metrics") if not metric.startswith("so") and not 'accum' in metric and not 'total' in metric and not 'standalone' in metric}
      metrics.update(metrics_E1)
      # Einf_metrics = ["train_lossFD", "train_loss_pct"]
      Einf_metrics = []
      metrics_Einf = {metric+"Einf": {arch.tostr():SumOfWhatever(measurements=metrics[metric][arch.tostr()], e=100).get_time_series(chunked=True) for arch in archs} for metric,v in tqdm(metrics.items(), desc = "Calculating Einf metrics") if metric in Einf_metrics and not metric.startswith("so") and not 'accum' in metric and not 'total' in metric}
      metrics.update(metrics_Einf)      

    for key in metrics_FD.keys(): # Remove the pure FD metrics because they are useless anyways
      metrics.pop(key, None)
    
    start=time.time()
    corrs = {}
    to_logs = []
    # Need to only track TopXX performance for some metrics because NB301 is too slow
    core_metrics = ["sotl", "sotl_aug", "sovl", "sovalacc", "train_loss", "val_loss", "val_acc", "train_acc", "total_train", "total_val", "total_train_loss", "total_val_loss"]
    for idx, (k,v) in tqdm(enumerate(metrics.items()), desc="Calculating correlations", total = len(metrics)):
      if xargs.drop_fancy and k not in core_metrics:
        continue
      tqdm.write(f"Started computing correlations for {k}")

      if torch.is_tensor(v[next(iter(v.keys()))]):
        v = {inner_k: [[batch_elem.item() for batch_elem in epoch_list] for epoch_list in inner_v] for inner_k, inner_v in v.items()}
      # We cannot do logging synchronously with training becuase we need to know the results of all archs for i-th epoch before we can log correlations for that epoch
      # constant_metric = True if (k in total_metrics_keys or "upper" in k) else False
      constant_metric = True if ("upper" in k) else False      
      if len(archs) > 1:
        try:
          corr, to_log = calc_corrs_after_dfs(epochs=epochs, xloader=train_loader, steps_per_epoch=steps_per_epoch, metrics_depth_dim=v, 
            final_accs = final_accs, archs=archs, true_rankings = true_rankings, prefix=k, api=api, wandb_log=False, corrs_freq = xargs.corrs_freq, 
            constant=constant_metric, xargs=xargs, nth_tops = [1, 5, 10] if k in core_metrics else [1], 
            top_n_freq=1 if xargs.search_space_paper != "darts" else 100)
          corrs["corrs_"+k] = corr
          to_logs.append(to_log)
        except Exception as e:
          logger.log(f"Failed to compute corrs for {k} due to {e}")
          raise e

    arch_ranking_by_size = [{"arch":arch, "metric":metrics["total_arch_count"][arch][0][0]} for arch in metrics["total_arch_count"].keys()]
    arch_ranking_by_size = sorted(arch_ranking_by_size, key=lambda x: x["metric"], reverse=True)
    arch_true_rankings_by_size = {"cifar10":arch_ranking_by_size, "cifar100":arch_ranking_by_size,"cifar10-valid":arch_ranking_by_size, "ImageNet16-120":arch_ranking_by_size}
    for k in tqdm(["train_grad_accum", "train_lossE1", "sotl", "train_grad_mean_accum", "sogn"], desc = "Computing correlations for param counts"):
      # This calculates correlatiosn with parameter count (arch_param_count)
      if k not in metrics.keys():
        print(f"WARNING! Didnt find {k} in metrics keys: {list(metrics.keys())}")
        continue
      v = metrics[k]
      if torch.is_tensor(v[next(iter(v.keys()))]):
        v = {inner_k: [[batch_elem.item() for batch_elem in epoch_list] for epoch_list in inner_v] for inner_k, inner_v in v.items()}
      corr, to_log = calc_corrs_after_dfs(epochs=epochs, xloader=train_loader, steps_per_epoch=steps_per_epoch, metrics_depth_dim=v, 
        final_accs = final_accs, archs=archs, true_rankings = arch_true_rankings_by_size, corr_funs=None, prefix=k+"P", api=api, wandb_log=False, 
        corrs_freq = xargs.corrs_freq, constant=None, xargs=xargs)
      corrs["param_corrs_"+k] = corr
      to_logs.append(to_log) 

    print(f"Calc corrs time: {time.time()-start}")
    
    # Produces some charts to WANDB so that it is easier to see the distribution of accuracy of sampled architectures
    try:
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
    except Exception as e:
      logger.log(f"Arch charts failed due to {e}")
      
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

    try:
      wandb.log({"arch_perf":arch_perf_tables, "arch_perf_charts":arch_perf_charts})
    except Exception as e:
      logger.log(f"Logging WANDB charts failed due to {e}")

  # if style in ["sotl", "sovl"] and n_samples-start_arch_idx > 0: # otherwise, we are just reloading the previous checkpoint so should not save again
  #   corr_metrics_path = save_checkpoint({"metrics":original_metrics, "corrs": corrs, "train_stats": train_stats,
  #     "archs":archs, "start_arch_idx":arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},
  #     logger.path('corr_metrics'), logger, backup=False)

  #   print(f"Upload to WANDB at {corr_metrics_path.absolute()}")
  #   try:
  #     pass
  #     # wandb.save(str(corr_metrics_path.absolute()))
      
  #   except Exception as e:
  #     print(f"Upload to WANDB failed because {e}")

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



def main(xargs):
  import warnings # There are some PyTorch UserWarnings because of the gradient hacks later on
  warnings.filterwarnings("ignore", category=UserWarning)
  warnings.filterwarnings("ignore")

  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( max(int(xargs.workers), 1))
  if xargs.search_space_paper == "darts": # Need to maintain the original DARTS proxy design. Note that num_cells = 2 actually gives 2 + REDUCTION + 2 + REDUCTION + 2 total cells in the model definition
    assert xargs.num_cells == 2
    # assert xargs.max_nodes == 7
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
    config = config._replace(momentum = xargs.search_momentum)
  if xargs.search_lr_min is not None:
    config = config._replace(eta_min=xargs.search_lr_min)

  if os.environ.get("TORCH_WORKERS", None) is not None:
    dataloader_workers = int(os.environ["TORCH_WORKERS"])
  else:
    dataloader_workers = xargs.workers
  resolved_train_batch_size, resolved_val_batch_size = xargs.train_batch_size if xargs.train_batch_size is not None else config.batch_size, xargs.val_batch_size if xargs.val_batch_size is not None else config.test_batch_size
  # NOTE probably better idea to not use train_batch_size here to not accidentally change the supernet search?
  logger.log("Instantiating the Search loaders")
  search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', 
    (config.batch_size if xargs.search_batch_size is None else xargs.search_batch_size, config.test_batch_size if xargs.search_val_batch_size is None else xargs.search_val_batch_size), workers=dataloader_workers, 
    epochs=config.epochs + config.warmup, determinism=xargs.deterministic_loader, 
    merge_train_val = xargs.merge_train_val_supernet, merge_train_val_and_use_test = xargs.merge_train_val_and_use_test, 
    extra_split = xargs.cifar5m_split, valid_ratio=xargs.val_dset_ratio, use_only_train=xargs.use_only_train_supernet, xargs=xargs)
  logger.log("Instantiating the postnet loaders")
  train_data_postnet, valid_data_postnet, xshape_postnet, class_num_postnet = get_datasets(xargs.dataset_postnet, xargs.data_path, -1, mmap=xargs.mmap, total_samples=xargs.total_samples)
  search_loader_postnet, train_loader_postnet, valid_loader_postnet = get_nas_search_loaders(train_data_postnet, valid_data_postnet, xargs.dataset_postnet, 'configs/nas-benchmark/', 
    (resolved_train_batch_size, resolved_val_batch_size), workers=dataloader_workers, valid_ratio=xargs.val_dset_ratio, determinism=xargs.deterministic_loader, 
    epochs=xargs.eval_epochs, merge_train_val=xargs.merge_train_val_postnet, 
    merge_train_val_and_use_test = xargs.merge_train_val_and_use_test, extra_split = xargs.cifar5m_split, use_only_train=xargs.use_only_train_supernet, xargs=xargs)
  logger.log("Instantiating the stats loaders")
  _, train_loader_stats, val_loader_stats = get_nas_search_loaders(train_data_postnet, valid_data_postnet, xargs.dataset_postnet, 'configs/nas-benchmark/', 
    (512 if gpu_mem < 8147483648 else 1024, 512 if gpu_mem < 8147483648 else 1024), workers=dataloader_workers, valid_ratio=xargs.val_dset_ratio, determinism="all", 
    epochs=xargs.eval_epochs, merge_train_val=xargs.merge_train_val_postnet, 
    merge_train_val_and_use_test = xargs.merge_train_val_and_use_test, extra_split = xargs.cifar5m_split, use_only_train=xargs.use_only_train_supernet, xargs=xargs)
  logger.log(f"Using train batch size: {resolved_train_batch_size}, val batch size: {resolved_val_batch_size}")
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces(xargs.search_space, xargs.search_space_paper)

  if xargs.model_name is None:
    model_config = dict2config(
    dict(name='generic', super_type = "basic", C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes, num_classes=class_num,
          space=search_space, affine=bool(xargs.affine), track_running_stats=bool(xargs.track_running_stats)), None)
  elif xargs.model_name == "DARTS" or xargs.model_name == "generic_nasnet":
    super_type = "basic" if xargs.search_space in ["nats-bench", None] else "nasnet-super"
    model_config = dict2config(
        dict(name=xargs.model_name, super_type = super_type, C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes, 
        num_classes=class_num, stem_multiplier=3, multiplier=4, steps=4,
            space=search_space, affine=bool(xargs.affine), track_running_stats=bool(xargs.track_running_stats)), None)
  elif xargs.model_name == "nb101":
    # Not much from the config is used because most of it is just hardcoded into the model definition
    super_type = "nb101"
    model_config = dict2config(
      dict(name=xargs.model_name, super_type = super_type, C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes, 
      num_classes=class_num, stem_multiplier=3, multiplier=4, steps=4,
          space=search_space, affine=bool(xargs.affine), track_running_stats=bool(xargs.track_running_stats)), None)
  logger.log('search space : {:}'.format(search_space))
  logger.log('model config : {:}'.format(model_config))
  search_model = get_cell_based_tiny_net(model_config, xargs)
  search_model.set_algo(xargs.algo)
  search_model = search_model.cuda()

  # TODO this logging search model makes a big mess in the logs! And it is almost always the same anyways
  # logger.log('{:}'.format(search_model))
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.weights, config)
  a_optimizer = torch.optim.Adam(search_model.alphas, lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay, eps=xargs.arch_eps)
  if xargs.higher_params == "weights":
    if xargs.meta_optim == "adam":
      meta_optimizer = torch.optim.Adam(search_model.weights, lr=xargs.meta_lr, betas=(0.5, 0.999), weight_decay=xargs.meta_weight_decay, eps=xargs.arch_eps)
    elif xargs.meta_optim == "sgd":
      meta_optimizer = torch.optim.SGD(search_model.weights, lr=xargs.meta_lr, momentum = xargs.meta_momentum, weight_decay = xargs.meta_weight_decay)
    elif xargs.meta_optim == "arch" or xargs.meta_optim is None:
      logger.log(f"Meta optimizer is set equal to the arch optimizer since xargs.meta_optim={xargs.meta_optim}")
      meta_optimizer = a_optimizer
    else:
      raise NotImplementedError
    logger.log(f"Initialized meta optimizer {meta_optimizer} since higher_params={xargs.higher_params}")

  else:
    # assert xargs.algo != "random"
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
    if xargs.search_space_paper == "nats-bench":
      api = create(None, 'topology', fast_mode=True, verbose=False)
    elif xargs.search_space_paper == "darts":
      from utils.nb301 import NASBench301Wrapper
      api = NASBench301Wrapper()
    elif xargs.search_space_paper.startswith("nb101"):
      from utils.nb101 import NASBench101Wrapper

      api = NASBench101Wrapper(xargs=xargs)

  else:
    api = None
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  network, criterion = search_model, criterion.cuda()  # use a single GPU
  last_info_orig, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  arch_sampler = ArchSampler(api=api, model=network, mode=xargs.evenly_split, dataset=xargs.evenly_split_dset, op_names=network._op_names, 
                             max_nodes = xargs.max_nodes, search_space = xargs.search_space_paper)
  network.arch_sampler = arch_sampler # TODO this is kind of hacky.. might have to pass it in through instantation?
  network.xargs = xargs
  messed_up_checkpoint, greedynas_archs, baseline_search_logs, config_no_restart_cond = False, None, None, True

  if xargs.supernet_init_path is not None and not last_info_orig.exists():
    init_search_from_checkpoint(search_model, logger, xargs)
    network.set_cal_mode('urs', None)
    valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(train_loader, network, criterion, xargs.algo, logger, steps=8)
    network._mode = None
    logger.log(f"The loaded checkpoint has train loss: {valid_a_loss}, train acc: {valid_a_top1}")


  elif last_info_orig.exists() and not xargs.reinitialize: # automatically resume from previous checkpoint
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
        logger.log(f"Failed to load checkpoints due to {e} but will try to load backups now")
        try:
          last_info   = torch.load(os.fspath(last_info_orig)+"_backup")
          checkpoint  = torch.load(os.fspath(last_info['last_checkpoint'])+"_backup") 
        except Exception as e:
          logger.log(f"Failed to load checkpoint backups at last_info: {os.fspath(last_info_orig)+'_backup'}, checkpoint: {os.fspath(last_info['last_checkpoint'])+'_backup'}")
      
      if xargs.force_overwrite:
        logger.log("Testing initial search_func config to see if it can work")
        checkpoint_config = checkpoint["config"] if "config" in checkpoint.keys() else {"whatever":"bad"}
        cond1={k:v for k,v in vars(checkpoint_config).items() if ('path' not in k and 'dir' not in k and k not in ["dry_run", "workers", "mmap"])}
        cond2={k:v for k,v in vars(xargs).items() if ('path' not in k and 'dir' not in k and k not in ["dry_run", "workers", "mmap"])}
        logger.log(f"Checkpoint config: {cond1}")
        logger.log(f"Newly input config: {cond2}")
        different_items = {k: cond1[k] for k in cond1 if k in cond2 and cond1[k] != cond2[k]}
        config_no_restart_cond = (cond1 == cond2 and len(different_items) == 0)
        print(f"Config no restart cond: {config_no_restart_cond}, different_items={different_items}")
      
      if xargs.force_overwrite and not config_no_restart_cond:
        logger.log(f"Need to restart the checkpoint completely because force_overwrite={xargs.force_overwrite} and config_no_restart_cond={config_no_restart_cond}")
        raise NotImplementedError
      
      start_epoch, epoch = last_info['epoch'], last_info['epoch']
      genotypes, baseline   = checkpoint['genotypes'], checkpoint["baseline"]
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
      # baseline_search_logs = all_search_logs
      logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    except Exception as e:
      logger.log(f"Checkpoint got messed up and cannot be loaded due to {e}! Will have to restart")
      checkpoint, last_info = None, None
      messed_up_checkpoint = True

  if not (last_info_orig.exists() and not xargs.reinitialize and not (xargs.force_overwrite and not config_no_restart_cond)) or messed_up_checkpoint or (xargs.supernet_init_path is not None and not last_info_orig.exists()):
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
    if last_info_orig.exists() and not xargs.reinitialize and not (xargs.force_overwrite and not config_no_restart_cond): # automatically resume from previous checkpoint
      baseline_search_logs = all_search_logs # Search logs from the checkpoint we loaded previously
      logger.log("Need to reload checkpoint due to using extra supernet training")
      logger.log("=> loading extra checkpoint of the last-info '{:}' start".format(last_info_orig))
      if os.name == 'nt': # The last-info pickles have PosixPaths serialized in them, hence they cannot be instantied on Windows
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
      last_info   = torch.load(last_info_orig.resolve())
      start_epoch, epoch = last_info['epoch'], last_info['epoch']
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
    percentiles, supernets_decomposition, arch_groups_quartiles, archs_subset, grad_metrics_percs, metrics_factory, metrics_percs = init_supernets_decomposition(xargs, logger, checkpoint, network)
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
        evaled_metrics, evaled_sum_metrics = eval_archs_on_batch(xloader=cur_loader, archs = candidate_archs, network=network, criterion=criterion, same_batch=True, metric=xargs.greedynas_sampling, 
                                                                 train_steps=xargs.eval_archs_train_steps, train_loader=train_loader, w_optimizer=w_optimizer)
        best_archs = sorted(list(zip(candidate_archs, evaled_metrics)), key = lambda x: x[1]) # All metrics should be so that higher is better, and we sort in ascending (ie. best is last)
        logger.log(f"GreedyNAS archs are sampled greedily (candidate_num={xargs.eval_candidate_num}), head (arch_idx, metric)={[(api.archstr2index[arch_tuple[0].tostr()], arch_tuple[1]) for arch_tuple in best_archs[-10:]]}")
        greedynas_archs = [x[0] for x in best_archs[-xargs.eval_candidate_num:]]

    logger.log(f"Sampling architectures that will be used for GreedyNAS Supernet post-main-supernet training in search_func, head = {[api.archstr2index[x.tostr()] for x in greedynas_archs[0:10]]}")
  else:
    greedynas_archs = None
  supernet_key, replay_buffer = "supernet", None
  arch_perf_percs = {k:None for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
  valid_a_loss , valid_a_top1 , valid_a_top5 = 0, 0, 0 # Initialization because we do not store the losses in checkpoints
  for epoch in range(start_epoch if not xargs.reinitialize else 0, total_epoch + (xargs.greedynas_epochs if xargs.greedynas_epochs is not None else 0) if not xargs.reinitialize else 0):
    if epoch == total_epoch: # End of normal training, start of GreedyNAS
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
    if xargs.algo.startswith('gdas'):
      network.set_tau( xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (total_epoch-1) )
      logger.log('[RESET tau as : {:} and drop_path as {:}]'.format(network.tau, network.drop_path))
    if epoch < total_epoch: # Use all archs as usual in SPOS
      archs_to_sample_from = None
    elif epoch >= total_epoch and xargs.greedynas_epochs > 0:
      if epoch == total_epoch:
        logger.log(f"About to start GreedyNAS supernet training with archs(len={len(greedynas_archs)}), head={[api.archstr2index[x.tostr()] for x in greedynas_archs[0:10]]}")
      archs_to_sample_from = greedynas_archs
    
    if (xargs.w_warm_start is None or epoch >= xargs.w_warm_start) and not xargs.freeze_arch:
      search_w_loss, search_w_top1, search_w_top5, search_a_loss, search_a_top1, search_a_top5, supernet_metrics, supernet_metrics_by_arch, arch_overview, supernet_stds, eigenvalues \
                  = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, xargs.algo, logger, 
                    smoke_test=xargs.dry_run, api=api, epoch=epoch,
                    supernets_decomposition=supernets_decomposition, arch_groups_quartiles=arch_groups_quartiles, arch_groups_brackets=arch_groups_brackets,
                    all_archs=archs_to_sample_from, grad_metrics_percentiles=grad_metrics_percs, 
                    percentiles=percentiles, metrics_percs=metrics_percs, xargs=xargs, replay_buffer=replay_buffer, val_loader=valid_loader_postnet, train_loader=train_loader_postnet,
                    meta_optimizer=meta_optimizer)

    else:
      train_epoch(train_loader=train_loader, network=network, w_optimizer=w_optimizer, criterion=criterion, algo=xargs.algo, logger=logger)
      save_path = save_checkpoint({'epoch' : epoch + 1,
            'args'  : deepcopy(xargs),
            "config": deepcopy(xargs),
            'baseline'    : baseline,
            'search_model': search_model.state_dict(),
            'w_optimizer' : w_optimizer.state_dict(),
            'a_optimizer' : a_optimizer.state_dict(),
            'w_scheduler' : w_scheduler.state_dict(),
            'genotypes'   : [],
            'valid_accuracies' : [],
            "grad_metrics_percs" : [],
            "archs_subset" : [],
            "search_logs" : [],
            "search_sotl_stats": [],
            "greedynas_archs": []},
            model_base_path, logger, backup=False)
      continue
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

    # TODO PUT THIS BACK IN
    # for percentile in arch_perf_percs.keys(): # Finds a threshold for each performance bracket from the latest epoch so that we can do exploiting search later
    #   arch_perf_percs[percentile] = arch_overview["train_loss"][min(math.floor(len(arch_overview["train_loss"]) * (percentile/100)), len(arch_overview["train_loss"])-1)]
    # grad_log_keys = ["gn", "gnL1", "sogn", "sognL1", "grad_normalized", "grad_accum", "grad_accum_singleE", "grad_accum_decay", "grad_mean_accum", "grad_mean_sign", "grad_var_accum", "grad_var_decay_accum"]
    # if xargs.supernets_decomposition:
    #   for percentile in percentiles[1:]:
    #     for log_key in grad_log_keys:
    #       metrics_percs[supernet_key+"_"+log_key]["perc"+str(percentile)][epoch].append(grad_metrics_percs["perc"+str(percentile)]["supernet"][log_key])
      
    search_time.update(time.time() - start_time)
    logger.log('[{:}] search [base] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
    logger.log('[{:}] search [arch] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, search_a_loss, search_a_top1, search_a_top5))
    if xargs.algo == 'enas' and (xargs.train_controller_freq is None or (epoch % xargs.train_controller_freq == 0 or epoch == xargs.search_epochs - 1)):
      ctl_loss, ctl_acc, baseline, ctl_reward \
                                 = train_controller(valid_loader, network, criterion, a_optimizer, baseline, epoch_str, xargs.print_freq, logger, xargs, w_optimizer=w_optimizer, train_loader=train_loader)
      logger.log('[{:}] controller : loss={:}, acc={:}, baseline={:}, reward={:}'.format(epoch_str, ctl_loss, ctl_acc, baseline, ctl_reward))

    if (epoch % xargs.search_eval_freq == 0 or epoch == xargs.search_epochs - 1 or len(genotypes) == 0 or ('random' not in xargs.algo)) or epoch == total_epoch - 1:
      genotype, temp_accuracy = get_best_arch(train_loader, valid_loader, network, xargs.eval_candidate_num, xargs.algo, 
                                              xargs=xargs, criterion=criterion, logger=logger, api=api, search_epoch=epoch)
      logger.log('[{:}] - [get_best_arch] : {:} -> {:}'.format(epoch_str, genotype, temp_accuracy))
      valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion, xargs.algo, logger, steps=5)
      logger.log('[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}% | {:}'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5, genotype))
      genotypes[epoch] = genotype
      
    elif len(genotypes) > 0:
      genotype = genotypes[epoch-1]
      temp_accuracy = 0
    if xargs.algo.startswith('setn') or xargs.algo == 'enas':
      network.set_cal_mode('dynamic', genotype)
    elif xargs.algo.startswith('gdas'):
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
    logger.log(f"Querying genotype {genotypes[epoch]} at epoch={epoch}")
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

    per_epoch_to_log = {"search":{"train_loss":search_w_loss,  "train_loss_arch":search_a_loss, "train_acc":search_w_top1, "train_acc_arch":search_a_top1, "epoch":epoch, "eigval":eigenvalues, **supernet_stds,
      "final": summarize_results_by_dataset(genotypes[epoch], api=api, iepoch=199, hp='200')}, "ops": count_ops(genotypes[epoch])}
    search_to_log = per_epoch_to_log
    try:
      interim = {}
      for batch_idx in range(len(search_loader)):
        interim = {}
        for metric in supernet_metrics.keys():
          for bracket in supernet_metrics[metric].keys():
            interim[metric+"."+bracket] = supernet_metrics[metric][bracket][min(batch_idx, len(supernet_metrics[metric][bracket])-1)]

        search_to_log = {**search_to_log, **interim, "epoch":epoch, "batch":batch_idx, "true_step":epoch*len(search_loader)+batch_idx, **decomposition_logs}
        if batch_idx % xargs.search_logs_freq == 0 or batch_idx == len(search_loader) - 1:
          all_search_logs.append(search_to_log)
    except Exception as e:
      logger.log(f"""Failed to log per-bracket supernet searchs stats due to {e} at batch_idx={batch_idx}, metric={metric}, bracket={bracket},
         length of the supernet_metrics[metric][bracket] = {len(supernet_metrics[metric][bracket]) if bracket in supernet_metrics[metric] else 'bracket missing!'}""")
      if batch_idx % xargs.search_logs_freq == 0 or batch_idx == len(search_loader) - 1:
        all_search_logs.append(search_to_log)
    
    wandb.log(search_to_log) # Log it online and then rewrite later if necessary. But seeing it in real-time in WANDB is too useful to pass up on

    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    if epoch % xargs.checkpoint_freq == 0 or epoch == total_epoch-1 or epoch in [49, 99]:
      save_path = save_checkpoint({'epoch' : epoch + 1,
                  'args'  : deepcopy(xargs),
                  "config": deepcopy(xargs),
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
                  model_base_path, logger, backup=False)
      last_info = save_checkpoint({
            'epoch': epoch + 1,
            'args' : deepcopy(args),
            'last_checkpoint': save_path,
          }, logger.path('info'), logger, backup=False)
      if epoch == total_epoch - 1 and 'random' in xargs.algo:
        try:
          save_path = save_checkpoint({'epoch' : epoch + 1,
                    'args'  : deepcopy(xargs),
                    "config": deepcopy(xargs),
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
                    os.path.join(wandb.run.dir, model_base_path.name), logger, backup=False)
          last_info = save_checkpoint({
                'epoch': epoch + 1,
                'args' : deepcopy(args),
                'last_checkpoint': save_path,
              }, os.path.join(wandb.run.dir, logger.path('info').name), logger, backup=False)
        except Exception as e:
          logger.log(f"Failed to log final weights to WANDB run directory due to {e}")
        
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  if baseline_search_logs is not None:
    for search_log in tqdm(baseline_search_logs, desc = "Logging supernet search logs from the pretrained checkpoint"):
      wandb.log(search_log)
  else:
    logger.log("There are no pretrained search logs (in the sense that the supernet search would be initialized from a checkpoint)! Not logging anything")

  max_search_logs = 15000
  search_logs_iter = all_search_logs if len(all_search_logs) < max_search_logs else takespread(all_search_logs, max_search_logs) 
  for search_log in tqdm(search_logs_iter, desc = "Logging supernet search logs"):
    wandb.log(search_log)
  
  # the final post procedure : count the time
  start_time = time.time()

  if xargs.cand_eval_method in ['val_acc', 'val'] or "random" not in xargs.algo:
    genotype, temp_accuracy = get_best_arch(train_loader_postnet, valid_loader_postnet, network, xargs.eval_candidate_num, xargs.algo, xargs=xargs, 
                                            criterion=criterion, logger=logger, style="val", api=api, search_epoch=epoch, config=config)
  elif xargs.cand_eval_method == 'sotl': #TODO probably get rid of this
    if greedynas_archs is None: # TODO might want to implement some greedy sampling here? None will just sample uniformly as in SPOS
      logger.log("Since greedynas_archs=None, we will sample archs anew for get_best_arch")
      archs_to_sample_from = None
    else:
      archs_to_sample_from = greedynas_archs
      logger.log(f"Reusing greedynas_archs for get_best_arch with head = {[api.archstr2index[x.tostr()] for x in archs_to_sample_from]}")

    if xargs.finetune_search == "uniform":
      genotype, temp_accuracy = get_best_arch(train_loader_postnet, valid_loader_postnet, network, xargs.eval_candidate_num, xargs.algo, criterion=criterion, logger=logger, style=xargs.cand_eval_method, 
        w_optimizer=w_optimizer, w_scheduler=w_scheduler, config=config, epochs=xargs.eval_epochs, steps_per_epoch=xargs.steps_per_epoch, 
        api=api, additional_training = xargs.additional_training, val_loss_freq=xargs.val_loss_freq, 
        overwrite_additional_training=xargs.overwrite_additional_training, scheduler_type=xargs.scheduler, xargs=xargs, train_loader_stats=train_loader_stats, val_loader_stats=val_loader_stats, 
        model_config=model_config, all_archs=archs_to_sample_from, search_sotl_stats = search_sotl_stats)
    elif xargs.finetune_search == "rea":
      arch_mutator = mutate_topology_func(network._op_names)
      history, cur_best_arch, total_time = regularized_evolution_ws(network=network, train_loader=train_loader, population_size=xargs.rea_population, 
        sample_size=xargs.rea_sample, mutate_arch = arch_mutator, cycles=xargs.rea_cycles, arch_sampler=arch_sampler, api=api, 
        config=config, xargs=xargs, train_steps=xargs.steps_per_epoch, train_epochs = xargs.eval_epochs)
      genotype = cur_best_arch[-1]

  if xargs.algo.startswith('setn') or xargs.algo == 'enas':
    network.set_cal_mode('dynamic', genotype)
  elif xargs.algo.startswith('gdas'):
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
  wandb.log({"absolutely_final": results_by_dataset})
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
  parser.add_argument('--steps_per_epoch', type=lambda x: int(x) if x != "None" else None,           default=307,  help='Number of minibatches to train for when evaluating candidate architectures with SoTL')
  parser.add_argument('--eval_epochs',          type=int, default=1,   help='Number of epochs to train for when evaluating candidate architectures with SoTL')
  parser.add_argument('--additional_training',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=True,   help='Whether to train the supernet samples or just go through the training loop with no grads')
  parser.add_argument('--val_batch_size',          type=int, default=64,   help='Batch size for the val loader - this is crucial for SoVL and similar experiments, but bears no importance in the standard NASBench setup')
  parser.add_argument('--dry_run',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False,   help='WANDB dry run - whether to sync to the cloud')
  parser.add_argument('--val_dset_ratio',          type=float, default=1,   help='Only uses a ratio of X for the valid data loader. Used for testing SoValAcc robustness')
  parser.add_argument('--val_loss_freq',          type=int, default=1,   help='How often to calculate val loss during training. Probably better to only this for smoke tests as it is generally better to record all and then post-process if different results are desired')
  parser.add_argument('--overwrite_additional_training',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False,   help='Whether to load checkpoints of additional training')
  parser.add_argument('--scheduler',          type=str, default=None,   help='Whether to use different training protocol for the postnet training')
  parser.add_argument('--train_batch_size',          type=int, default=None,   help='Training batch size for the POST-SUPERNET TRAINING!')
  parser.add_argument('--lr',          type=float, default=0.001,   help='Constant LR for the POST-SUPERNET TRAINING!')
  parser.add_argument('--postnet_decay',          type=float, default=None,   help='Weight decay for the POST-SUPERNET TRAINING!')

  parser.add_argument('--deterministic_loader',          type=str, default='all', choices=['None', 'train', 'val', 'all'],   help='Whether to choose SequentialSampler or RandomSampler for data loaders')
  parser.add_argument('--reinitialize',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Whether to use trained supernetwork weights for initialization')
  parser.add_argument('--individual_logs',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Whether to log each of the eval_candidate_num sampled architectures as a separate WANDB run')
  parser.add_argument('--total_estimator_steps',          type=int, default=5, help='Number of batches for evaluating the total_val/total_train etc. metrics')
  parser.add_argument('--corrs_freq',          type=int, default=4, help='Calculate corrs based on every i-th minibatch')
  parser.add_argument('--mmap',          type=str, default=None, help='Whether to mmap cifar5m')
  parser.add_argument('--search_epochs',          type=int, default=None, help='Can be used to explicitly set the number of search epochs')
  parser.add_argument('--size_percentile',          type=float, default=None, help='Percentile of arch param count in NASBench sampling, ie. 0.9 will give top 10% archs by param count only')
  parser.add_argument('--total_samples',          type=int, default=None, help='Number of total samples in dataset. Useful for limiting Cifar5m')
  parser.add_argument('--restart',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=None, help='WHether to force or disable restart of training via must_restart')
  parser.add_argument('--grads_analysis',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='WHether to force or disable restart of training via must_restart')
  parser.add_argument('--perf_percentile',          type=float, default=None, help='Perf percentile of architectures to sample from')
  parser.add_argument('--resample',          type=str, default=False, help='Only makes sense when also using reinitialize')
  parser.add_argument('--supernets_decomposition',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Track updates to supernetwork by quartile')
  parser.add_argument('--supernets_decomposition_mode',          type=str, choices=["perf", "size"], default="perf", help='Track updates to supernetwork by quartile')
  parser.add_argument('--supernets_decomposition_topk',          type=int, default=-1, help='How many archs to sample from the search space')
  parser.add_argument('--evenify_training',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Since subnetworks might come out unevenly trained, we can set a standard number of epochs-equivalent-of-trianing-from-scratch and match that for each')
  parser.add_argument('--adaptive_lr',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else x, choices=["custom", "1cycle"], default=False, help='Do a quick search for best LR before post-supernet training')
  parser.add_argument('--sandwich',          type=int, default=None, help='Do a quick search for best LR before post-supernet training')
  parser.add_argument('--sandwich_mode',          type=str, default=None, choices=["fairnas", "quartiles", None], help='Special sampling like size quartiles/FairNAS etc.')
  parser.add_argument('--sandwich_computation',          type=str, default="serial", choices=["serial", "parallel"], help='Do a quick search for best LR before post-supernet training')

  parser.add_argument('--force_overwrite',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Load saved seed or not')
  parser.add_argument('--greedynas_epochs',          type=int, default=None, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--greedynas_lr',          type=float, default=0.01, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--greedynas_sampling',          type=str, default="random", choices=["random", "acc", "loss"], help='Metric to sample the GreedyNAS architectures for supernet finetuning')
  parser.add_argument('--greedynas_sampling_loader',          type=str, default="train", choices=["train", "val"], help='The dataset to evaluate GreedyNAS archs on')
  parser.add_argument('--greedynas_candidate_num',          type=int, default=1000, help='The number of cand archs to evaluate for picking the best ones in GreedyNAS sampling')

  parser.add_argument('--merge_train_val_postnet',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--merge_train_val_supernet',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--postnet_switch_train_val',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--use_only_train_supernet',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')

  parser.add_argument('--dataset_postnet',          type=str, default=None, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'cifar5m'], help='Whether to do additional supernetwork SPOS training but using only the archs that are to be selected for short training later')
  parser.add_argument('--replay_buffer',          type=int, default=None, help='Replay buffer to tackle multi-model forgetting')
  parser.add_argument('--replay_buffer_mode',          type=str, default="random", choices=["random", "perf", "size", None], help='How to figure out what to put in the replay buffer')
  parser.add_argument('--replay_buffer_percentile',          type=float, default=0.9, help='Replay buffer percentile of performance etc.')
  parser.add_argument('--replay_buffer_weight',          type=float, default=0.5, help='Trade off between new arch loss and buffer loss')
  parser.add_argument('--replay_buffer_metric',          type=str, default="train_loss", choices=["train_loss", "train_acc", "val_acc", "val_loss"], help='Trade off between new arch loss and buffer loss')
  parser.add_argument('--evenly_split',          type=str, default=None, choices=["perf", "size"], help='Whether to split the NASBench archs into eval_candidate_num brackets and then take an arch from each bracket to ensure they are not too similar')
  parser.add_argument('--evenly_split_dset',          type=str, default=None, choices=["all", "cifar10", "cifar100", "ImageNet16-120"], help='Whether to split the NASBench archs into eval_candidate_num brackets and then take an arch from each bracket to ensure they are not too similar')
  parser.add_argument('--merge_train_val_and_use_test',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='Merges CIFAR10 train/val into one (ie. not split in half) AND then also treats test set as validation')
  parser.add_argument('--search_batch_size',          type=int, default=None, help='Controls batch size for the supernet training (search/GreedyNAS finetune phase)')
  parser.add_argument('--search_val_batch_size',          type=int, default=None, help='Controls batch size for the supernet training (search/GreedyNAS finetune phase)')

  parser.add_argument('--search_eval_freq',          type=int, default=5, help='How often to run get_best_arch during supernet training')
  parser.add_argument('--search_lr',          type=float, default=None, help='LR for teh superneat search training')
  parser.add_argument('--search_momentum',          type=float, default=None, help='Momentum in the supernet search training')
  parser.add_argument('--overwrite_supernet_finetuning',          type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=True, help='Whether to load additional checkpoints on top of the normal training -')
  parser.add_argument('--eval_arch_train_steps',          type=int, default=None, help='Whether to load additional checkpoints on top of the normal training -')
  parser.add_argument('--supernet_init_path' ,       type=str,   default=None, help='The path of pretrained checkpoint')
  parser.add_argument('--metaprox_lambda' ,       type=float,   default=0.1, help='Number of adaptation steps in MetaProx')
  parser.add_argument('--search_space_paper' ,       type=str,   default="nats-bench", choices=["darts", "nats-bench", "nb101_1", "nb101_2", "nb101_3"], help='Number of adaptation steps in MetaProx')
  parser.add_argument('--checkpoint_freq' ,       type=int,   default=3, help='How often to pickle checkpoints')
  
  parser.add_argument('--higher_method' ,       type=str, choices=['val', 'sotl', "val_multiple", "val_multiple_v2", "sotl_v2"],   default='val', help='Whether to take meta gradients with respect to SoTL or val set (which might be the same as training set if they were merged)')
  parser.add_argument('--higher_params' ,       type=str, choices=['weights', 'arch'],   default='weights', help='Whether to do meta-gradients with respect to the meta-weights or architecture')
  parser.add_argument('--higher_order' ,       type=str, choices=['first', 'second', None],   default=None, help='Whether to do meta-gradients with respect to the meta-weights or architecture')
  parser.add_argument('--higher_loop' ,       type=str, choices=['bilevel', 'joint'],   default=None, help='Whether to make a copy of network for the Higher rollout or not. If we do not copy, it will be as in joint training')
  parser.add_argument('--higher_reduction' ,       type=str, choices=['mean', 'sum'],   default='sum', help='Reduction across inner steps - relevant for first-order approximation')
  parser.add_argument('--higher_reduction_outer' ,       type=str, choices=['mean', 'sum'],   default='sum', help='Reduction across the meta-betach size')
  parser.add_argument('--higher_loop_joint_steps' ,       type=int, default=None, help='Useful for making inequal steps of adaptation for weights and arch when doing SOTL grads')

  parser.add_argument('--arch_warm_start' ,       type=int, default=None, help='How long to train only weights without arch updates')

  parser.add_argument('--first_order_strategy' ,       type=str, choices=['last', 'every'],   default='every', help='Whether to make a copy of network for the Higher rollout or not. If we do not copy, it will be as in joint training')

  parser.add_argument('--meta_algo' ,       type=str, choices=['reptile', 'metaprox', "reptile_higher", "metaprox_higher", 'darts_higher', "gdas_higher", "setn_higher", "enas_higher", "maml", "maml_higher", "fomaml", "fomaml_higher"],   default=None, help='Whether to do meta-gradients with respect to the meta-weights or architecture')
  
  parser.add_argument('--inner_steps' ,       type=int,   default=None, help='Number of steps to do in the inner loop of bilevel meta-learning')
  parser.add_argument('--inner_steps_same_batch' ,       type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,   default=True, help='Number of steps to do in the inner loop of bilevel meta-learning')
  parser.add_argument('--inner_sandwich',          type=int, default=None, help='Number of sampled archs for inner multipath. Useful for Reptile where multi-path is taken as a single model')

  parser.add_argument('--hessian' ,       type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,   default=True, help='Whether to track eigenspectrum in DARTS')

  parser.add_argument('--meta_optim' ,       type=str,   default='sgd', choices=['sgd', 'adam', 'arch', None], help='Kind of meta optimizer')
  parser.add_argument('--meta_lr' ,       type=float,   default=1, help='Meta optimizer LR. Can be considered as the interpolation coefficient for Reptile/Metaprox')
  parser.add_argument('--meta_momentum' ,       type=float,   default=0.0, help='Meta optimizer SGD momentum (if applicable). In practice, Reptile works like absolute garbage with non-zero momentum')
  parser.add_argument('--meta_weight_decay' ,       type=float,   default=0.0, help='Meta optimizer SGD weight decay (if applicable)')
  parser.add_argument('--cos_restart_len' ,       type=int,   default=None, help='Meta optimizer SGD momentum (if applicable)')
  parser.add_argument('--cifar5m_split' ,       type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,   default=True, help='Whether to split Cifar5M into multiple chunks that represent epochs; setting to True will make it like synthetic CIFAR10. Otherwise it will be a very large 1 epoch dataset')
  
  parser.add_argument('--finetune_search' ,       type=str,   default="uniform", choices=["uniform", "rea"], help='Sample size in each cycle of REA')
  parser.add_argument('--rea_metric' ,       type=str,   default="sotl", help='Whether to split Cifar5M into multiple chunks so that each epoch never repeats the same data twice; setting to True will make it like synthetic CIFAR10')
  parser.add_argument('--rea_sample' ,       type=int,   default=3, help='Sample size in each cycle of REA')
  parser.add_argument('--rea_population' ,       type=int,   default=10, help='Sample size in each cycle of REA')
  parser.add_argument('--rea_cycles' ,       type=int,   default=None, help='How many cycles of REA to run')
  parser.add_argument('--rea_epochs' ,       type=int,   default=100, help='Total epoch budget for REA')
  parser.add_argument('--model_name' ,       type=str,   default=None, choices=[None, "DARTS", "GDAS", "generic", "generic_nasnet", "nb101"], help='Picking the right model to instantiate. For DARTS, we need to have the two different normal/reduction cells which are not in the generic NAS201 model')
  parser.add_argument('--drop_fancy' ,       type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,   default=False, help='Drop special metrics in get_best_arch to make the finetuning proceed faster')
  parser.add_argument('--archs_split' ,       type=lambda x: None if x in ["False", "false", "", "None", False, None] else str(x),   default="default", help='Drop special metrics in get_best_arch to make the finetuning proceed faster')
  parser.add_argument('--save_archs_split' ,       type=str,   default=None, help='Drop special metrics in get_best_arch to make the finetuning proceed faster')
  parser.add_argument('--save_train_split' ,       type=str,   default=None, help='Save train split somewhere')
  parser.add_argument('--train_split' ,       type=str,   default=None, help='Load train split somewhere')

  parser.add_argument('--implicit_algo' ,       type=str,   default=None, choices=['cg', 'neumann'], help='Drop special metrics in get_best_arch to make the finetuning proceed faster')
  parser.add_argument('--implicit_steps' ,       type=int,   default=20, help='Number of steps in CG/Neumann appproximation')
  parser.add_argument('--w_warm_start' ,       type=int,   default=None, help='Dont train architecture for the first X epochs')
  parser.add_argument('--implicit_grad_clip' ,       type=lambda x: int(x) if x not in [None, "None"] else None,   default=100, help='Number of steps in CG/Neumann appproximation')
  parser.add_argument('--steps_per_epoch_supernet' ,       type=int,   default=None, help='Drop special metrics in get_best_arch to make the finetuning proceed faster')

  parser.add_argument('--debug' ,       type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,   default=None, help='Drop special metrics in get_best_arch to make the finetuning proceed faster')
  parser.add_argument('--cifar100_merge_all' ,       type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,   default=None, help='Drop special metrics in get_best_arch to make the finetuning proceed faster')
  parser.add_argument('--freeze_arch' ,       type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,   default=None, help='Train only weights and not arch - useful for DARTS pretraining without searching, for instance')
  parser.add_argument('--search_logs_freq' ,       type=int,   default=25, help='Train only weights and not arch - useful for DARTS pretraining without searching, for instance')

  parser.add_argument('--discrete_diffnas_method' ,       type=str,   default="val", help='Whether to use Val or SOTL-ish metrics as reward in GDAS/ENAS/..')
  parser.add_argument('--discrete_diffnas_steps' ,       type=int,   default=5, help='How many finetuning steps to do to collect SOTL-ish metrics in GDAS/ENAS/... Applicalbe only when using discrete_diffnas_method=sotl')
  
  parser.add_argument('--search_lr_min' ,       type=float,   default=None, help='Min LR to converge to in the search phase')
  parser.add_argument('--always_refresh_arch_oneshot' ,       type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,   default=False, help='Determines behavior of GDAS in inner loop. If true, will sample a new arch on every inner step, otherwise use the same for the whole unrolling')
  parser.add_argument('--bilevel_train_steps' ,       type=int,   default=None, help='Can be used to have asymmetry in the unrolling length vs training for real length')
  parser.add_argument('--bilevel_refresh_arch' ,       type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,   default=None, help='Refresh arch during bilevel train-for-real phase. Useful for GDAS')
  parser.add_argument('--train_controller_freq' ,       type=int,   default=None, help='Refresh arch during bilevel train-for-real phase. Useful for GDAS')
  parser.add_argument('--grad_drop_p' ,       type=float,   default=0.0, help='Probability of dropping weights gradients with GradDrop')


  args = parser.parse_args()

  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'
  mp.set_start_method('spawn')
  wandb_auth()
  run = wandb.init(project="NAS", group=f"Search_Cell_{args.algo}", reinit=True)

  if args.archs_split == "default" and args.algo == "random":
    if args.search_space_paper == "nats-bench":
      args.archs_split = "archs_random_100_seed50.pkl"
      args.eval_candidate_num = 100
      print(f"Changed archs_split={args.archs_split} and eval_candidate_num={args.eval_candidate_num}")

    elif args.search_space_paper == "darts":
      args.archs_split = "archs_darts_random_350_seed1000.pkl"
      args.eval_candidate_num = 350
      print(f"Changed archs_split={args.archs_split} and eval_candidate_num={args.eval_candidate_num}")
    elif args.search_space_paper == "nb101_1":
      args.archs_split = "archs_nb101_1_random_200_seed3000.pkl"
      args.eval_candidate_num = 200
    elif args.search_space_paper == "nb101_2":
      args.archs_split = "archs_nb101_2_random_200_seed3000.pkl"
      args.eval_candidate_num = 200
    elif args.search_space_paper == "nb101_3":
      args.archs_split = "archs_nb101_3_random_200_seed3000.pkl"
      args.eval_candidate_num = 200
    # args.archs_split = None
    else:
      raise NotImplementedError 
    
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