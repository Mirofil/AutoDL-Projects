
import os, sys, time, random, argparse, math
import numpy as np
from copy import deepcopy
from collections import defaultdict
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from pathlib import Path
from hessian_eigenthings import compute_hessian_eigenthings

lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from log_utils    import AverageMeter, time_string, convert_secs2time
from utils        import count_parameters_in_MB, obtain_accuracy
from utils.sotl_utils import _hessian, analyze_grads, eval_archs_on_batch
from typing import *
from models.cell_searchs.generic_model import ArchSampler


def sample_arch_and_set_mode_search(args, outer_iter, sampled_archs, api, network, algo, arch_sampler, 
                                    step, logger, epoch, supernets_decomposition, all_archs, arch_groups_brackets, placement=None):
    parsed_algo = algo.split("_")
    sampling_done, lowest_loss_arch, lowest_loss = False, None, 10000 # Used for GreedyNAS online search space pruning - might have to resample many times until we find an architecture below the required threshold
    sampled_arch = None
    branch = None
    if algo.startswith('setn'):
        branch = "setn"
        sampled_arch = network.dync_genotype(True)
        network.set_cal_mode('dynamic', sampled_arch)
    elif algo.startswith('gdas'):
        branch = "gdas"
        network.set_cal_mode('gdas', None)
        if sampled_archs is not None and (not args.refresh_arch_oneshot or (args.refresh_arch_oneshot == "train_real" and placement in ["inner_sandwich", "outer"])):
          assert placement in ["inner_sandwich", "outer", None]
          network.last_gumbels = sampled_archs[outer_iter]
          network.refresh_arch_oneshot = False
          if epoch < 2 and step < 3:
            logger.log(f"Set Gumbels at epoch={epoch}, outer_iter={outer_iter} = {network.last_gumbels}")
        sampled_arch = network.genotype
    elif algo.startswith('darts'):
        branch = "darts"
        network.set_cal_mode('joint', None)
        sampled_arch = network.genotype
    elif "random_" in algo and len(parsed_algo) > 1 and ("perf" in algo or "size" in algo):
        branch = "random1"
        if args.search_space_paper == "nats-bench":
            sampled_arch = arch_sampler.sample()[0]
            network.set_cal_mode('dynamic', sampled_arch)
        else:
            network.set_cal_mode('urs')
    elif "random" in algo and args.evenly_split is not None: # TODO should just sample outside of the function and pass it in as all_archs?
        branch = "random2"
        sampled_arch = arch_sampler.sample(mode="evenly_split", candidate_num = args.eval_candidate_num)[0]
        network.set_cal_mode('dynamic', sampled_arch)

    elif "random" in algo and args.sandwich is not None and args.sandwich > 1:
        branch = "random_quartiles"
        if args.search_space_paper == "nats-bench":
            assert args.sandwich == 4 or args.sandwich_mode != "quartiles" # 4 corresponds to using quartiles
            if step < 2 and epoch is not None and epoch < 2:
                logger.log(f"Sampling from the Sandwich branch with sandwich={args.sandwich} and sandwich_mode={args.sandwich_mode}")
                logger.log(f"Sampled archs = {[api.archstr2index[x.tostr()] for x in sampled_archs]}, cur arch = {sampled_archs[outer_iter]}")
            sampled_arch = sampled_archs[outer_iter] # Pick the corresponding quartile architecture for this iteration
            network.set_cal_mode('dynamic', sampled_arch)
        else:
          sampled_arch = sampled_archs[outer_iter]
          network.set_cal_mode('dynamic', sampled_arch)
    elif "random" in algo and args.sandwich is not None and args.sandwich > 1 and args.sandwich_mode == "fairnas":
        branch = "random_fairnas"
        assert args.sandwich == len(network._op_names)
        sampled_arch = sampled_archs[outer_iter] # Pick the corresponding quartile architecture for this iteration
        if step < 2 and epoch is not None and epoch < 2:
            logger.log(f"Sampling from the FairNAS branch with sandwich={args.sandwich} and sandwich_mode={args.sandwich_mode}, arch={sampled_arch}")
            logger.log(f"Sampled archs = {[api.archstr2index[x.tostr()] for x in sampled_archs]}, cur arch = {sampled_archs[outer_iter]}")

        network.set_cal_mode('dynamic', sampled_arch)
    elif "random_" in algo and "grad" in algo:
        network.set_cal_mode('urs')
    elif algo == 'random': # NOTE the original branch needs to be last so that it is fall-through for all the special 'random' branches
        branch = "random"
        if supernets_decomposition or all_archs is not None or arch_groups_brackets is not None:
            branch = "random_weird"
            if all_archs is not None:
                sampled_arch = random.sample(all_archs, 1)[0]
                network.set_cal_mode('dynamic', sampled_arch)
            else:
                if args.search_space_paper == "nats-bench":
                    sampled_arch = arch_sampler.sample(mode="random")[0]
                    network.set_cal_mode('dynamic', sampled_arch)
                else:
                  sampled_arch = network.sample_arch()
                  network.set_cal_mode('dynamic', sampled_arch)
        else:
          if args.search_space_paper == "nats-bench":
              branch="random_basic"
              network.set_cal_mode('urs', None)
          else:
            sampled_arch = network.sample_arch()
            network.set_cal_mode('dynamic', sampled_arch)
    elif algo == 'enas':
        with torch.no_grad():
            network.controller.eval()
            _, _, sampled_arch = network.controller()
            network.set_cal_mode('dynamic', sampled_arch)
    else:
        raise ValueError('Invalid algo name : {:}'.format(algo))
      
    # if step < 2:
    #   print(f"Sample_arch through branch={branch}")
    return sampled_arch


def sample_new_arch(network, algo, arch_sampler, sandwich_archs, all_archs, base_inputs, base_targets, arch_overview, loss_threshold, outer_iter, step, logger, supernets_decomposition, arch_groups_brackets, args):
# Need to sample a new architecture (considering it as a meta-batch dimension)
    parsed_algo = algo.split("_")
    sampling_done = False # Used for GreedyNAS online search space pruning - might have to resample many times until we find an architecture below the required threshold
    lowest_loss_arch = None
    lowest_loss = 10000
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
    return sampled_arch

def format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, args, loader_type="train-val"):

    # base_inputs, arch_inputs = base_inputs.cuda(non_blocking=True), arch_inputs.cuda(non_blocking=True)
    # base_targets, arch_targets = base_targets.cuda(non_blocking=True), arch_targets.cuda(non_blocking=True)
    
    base_inputs, base_targets = base_inputs.cuda(non_blocking=True), base_targets.cuda(non_blocking=True)
    arch_inputs, arch_targets = arch_inputs.cuda(non_blocking=True), arch_targets.cuda(non_blocking=True)
    if args.higher_method == "sotl":
        arch_inputs, arch_targets = None, None
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = [base_inputs], [base_targets], [arch_inputs], [arch_targets]
    for extra_step in range(inner_steps-1):
        if args.inner_steps_same_batch:
            all_base_inputs.append(base_inputs)
            all_base_targets.append(base_targets)
            all_arch_inputs.append(arch_inputs)
            all_arch_targets.append(arch_targets)
            continue # If using the same batch, we should not try to query the search_loader_iter for more samples
        try:
            if loader_type == "train-val" or loader_type == "train-train":
              extra_base_inputs, extra_base_targets, extra_arch_inputs, extra_arch_targets = next(search_loader_iter)
            else:
              extra_base_inputs, extra_base_targets = next(search_loader_iter)
              extra_arch_inputs, extra_arch_targets = None, None
        except:
            continue
        # extra_base_inputs, extra_arch_inputs = extra_base_inputs.cuda(non_blocking=True), extra_arch_inputs.cuda(non_blocking=True)
        # extra_base_targets, extra_arch_targets = extra_base_targets.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        
        extra_base_inputs, extra_base_targets = extra_base_inputs.cuda(non_blocking=True), extra_base_targets.cuda(non_blocking=True)
        if extra_arch_inputs is not None and extra_arch_targets is not None:
          extra_arch_inputs, extra_arch_targets = extra_arch_inputs.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        
        all_base_inputs.append(extra_base_inputs)
        all_base_targets.append(extra_base_targets)
        all_arch_inputs.append(extra_arch_inputs)
        all_arch_targets.append(extra_arch_targets)

    return all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets


def update_brackets(supernet_train_stats_by_arch, supernet_train_stats, supernet_train_stats_avgmeters, arch_groups_brackets, arch_overview, items, all_brackets, sampled_arch, args):
    if arch_overview["cur_arch"] is not None:
      if type(arch_groups_brackets) is dict:
          cur_bracket = arch_groups_brackets[arch_overview["cur_arch"].tostr()]
          for key, val in items:
              supernet_train_stats_by_arch[sampled_arch.tostr()][key].append(val)
              for bracket in all_brackets:
                  if bracket == cur_bracket:
                      supernet_train_stats[key]["sup"+str(cur_bracket)].append(val)
                      supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(cur_bracket)].update(val)
                      supernet_train_stats[key+"AVG"]["sup"+str(cur_bracket)].append(supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(cur_bracket)].avg)
                  else:
                      item_to_add = supernet_train_stats[key]["sup"+str(bracket)][-1] if len(supernet_train_stats[key]["sup"+str(bracket)]) > 0 else 3.14159
                      supernet_train_stats[key]["sup"+str(bracket)].append(item_to_add)
                      avg_to_add = supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(bracket)].avg if supernet_train_stats_avgmeters[key+"AVG"]["sup"+str(bracket)].avg > 0 else 3.14159
                      supernet_train_stats[key+"AVG"]["sup"+str(bracket)].append(avg_to_add)

def get_finetune_scheduler(scheduler_type, config, xargs, network2, epochs=None, logger=None, best_lr=None):

    if scheduler_type in ['linear_warmup', 'linear']:
        config = config._replace(scheduler=scheduler_type, warmup=1, eta_min=0, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    elif scheduler_type == "cos_reinit":
        # In practice, this leads to constant LR = 0.025 since the original Cosine LR is annealed over 100 epochs and our training schedule is very short
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    elif scheduler_type == "cos_restarts":
        config = config._replace(scheduler='cos_restarts', warmup=0, epochs=epochs, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config, xargs)
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
        config = config._replace(scheduler='constant', constant_lr=xargs.lr if not xargs.adaptive_lr else best_lr, 
                                 decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    elif scheduler_type == "constant":
        config = config._replace(scheduler='constant', constant_lr=xargs.lr if not xargs.adaptive_lr else best_lr, decay = 0.0005 if xargs.postnet_decay is None else xargs.postnet_decay)
        w_optimizer2, w_scheduler2, criterion = get_optim_scheduler(network2.weights, config)
    else:
        print(f"Unrecognized scheduler at {scheduler_type}")
        raise NotImplementedError
    return w_optimizer2, w_scheduler2, criterion

def find_best_lr(xargs, network2, train_loader, config, arch_idx):
  lr_counts = {}
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
      print(lr_results)
      lr_counts[best_lr] += 1
  else:
    best_lr = None
    
  return best_lr
    
    
def sample_arch_and_set_mode(network, algo, arch_sampler, all_archs, parsed_algo, args, step, logger, sampled_archs, outer_iter):
    sampled_arch = None
    if algo.startswith('setn'):
        sampled_arch = network.dync_genotype(True)
        network.set_cal_mode('dynamic', sampled_arch)
    elif algo.startswith('gdas'):
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

    elif "random" in algo and args.sandwich is not None and args.sandwich > 1 and args.sandwich_mode == "quartiles":
        if args.search_space_paper == "nats-bench":
            assert args.sandwich == 4 # 4 corresponds to using quartiles
            if step == 0:
                logger.log(f"Sampling from the Sandwich branch with sandwich={args.sandwich} and sandwich_mode={args.sandwich_mode}")
            sampled_arch = sampled_archs[outer_iter] # Pick the corresponding quartile architecture for this iteration
            network.set_cal_mode('dynamic', sampled_arch)
        else:
            network.set_cal_mode('urs')
    elif "random" in algo and args.sandwich is not None and args.sandwich > 1 and args.sandwich_mode == "fairnas":
        assert args.sandwich == len(network._op_names)
        sampled_arch = sampled_archs[outer_iter] # Pick the corresponding quartile architecture for this iteration
        if step == 0:
            logger.log(f"Sampling from the FairNAS branch with sandwich={args.sandwich} and sandwich_mode={args.sandwich_mode}, arch={sampled_arch}")
        network.set_cal_mode('dynamic', sampled_arch)
    elif "random_" in algo and "grad" in algo:
        network.set_cal_mode('urs')
    elif algo == 'random': # NOTE the original branch needs to be last so that it is fall-through for all the special 'random' branches
        if all_archs is not None:
            sampled_arch = random.sample(all_archs, 1)[0]
            network.set_cal_mode('dynamic', sampled_arch)
        else:
            if args.search_space_paper == "nats-bench":
                sampled_arch = arch_sampler.sample(mode="random")[0]
                network.set_cal_mode('dynamic', sampled_arch)
            else:
                sampled_arch = network.sample_arch()
                network.set_cal_mode('dynamic', sampled_arch)
    elif algo == 'enas':
        with torch.no_grad():
            network.controller.eval()
            _, _, sampled_arch = network.controller()
            network.set_cal_mode('dynamic', sampled_arch)
    else:
        raise ValueError('Invalid algo name : {:}'.format(algo))
    return sampled_arch

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

  network.train()
  return loss.avg, top1.avg, top5.avg


def train_controller(xloader, network, criterion, optimizer, prev_baseline, epoch_str, print_freq, logger, xargs, w_optimizer=None, train_loader=None):
  # config. (containing some necessary arg)
  #   baseline: The baseline score (i.e. average val_acc) from the previous epoch
  # NOTE the xloader is typically val loader
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
  for step in tqdm(range(controller_train_steps * controller_num_aggregate), desc = "Training controller", total=controller_train_steps*controller_num_aggregate):
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
    if xargs.discrete_diffnas_method in [None, "val"]:
      with torch.no_grad():
        network.set_cal_mode('dynamic', sampled_arch)
        _, logits = network(inputs)
        loss = criterion(logits, targets)
        reward_metric, val_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        reward_metric  = reward_metric.view(-1) / 100
    elif xargs.discrete_diffnas_method in ["sotl"]:
      if step == 0: print(f"ENAS train controller - supernet weight sample before finetune: {str(list(network.parameters())[1])[0:80]}")
      eval_metrics, finetune_metrics = eval_archs_on_batch(xloader=xloader, archs=[sampled_arch], network=network, criterion=criterion, metric="loss", 
                                                           train_steps=xargs.discrete_diffnas_steps, w_optimizer=w_optimizer, train_loader=train_loader, 
                                                           progress_bar=False)
      if step == 0: print(f"ENAS train controller - supernet weight sample after finetune (should be the same to make sure we do not change the original network): {str(list(network.parameters())[1])[0:80]}")
      reward_metric = torch.tensor(finetune_metrics[sampled_arch]["sotl"][-1]) # Take the SOTL over all training steps as the reward
    else:
      raise NotImplementedError
        
        
    reward = reward_metric + controller_entropy_weight * entropy
    if prev_baseline is None:
      baseline = reward_metric
    else:
      baseline = prev_baseline - (1 - controller_bl_dec) * (prev_baseline - reward)
   
    loss = -1 * log_prob * (reward - baseline)
    
    # account
    RewardMeter.update(reward.item())
    BaselineMeter.update(baseline.item())
    ValAccMeter.update(reward_metric.item()*100)
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


def regularized_evolution_ws(network, train_loader, population_size, sample_size, mutate_arch, cycles, arch_sampler, api, config, xargs, train_steps=15, train_epochs=1, metric="loss"):
  """Algorithm for regularized evolution (i.e. aging evolution).
  
  Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
  Classifier Architecture Search".
  
  Args:
    cycles: the number of cycles the algorithm should run for.
    population_size: the number of individuals to keep in the population.
    sample_size: the number of individuals that should participate in each tournament.
    time_budget: the upper bound of searching cost

  Returns:
    history: a list of `Model` instances, representing all the models computed
        during the evolution experiment.
  """
  # init_model = deepcopy(network.state_dict())
  # init_optim = deepcopy(w_optimizer.state_dict())

  population = collections.deque()
  api.reset_time()
  history = [] # Not used by the algorithm, only used to report results.
  cur_best_arch = []
  stats = {"pop":{"mean":[], "std":[]}}
  top_ns = [1, 5, 10]
  total_time = 0
  model_init = deepcopy(network)

  cycle_len = train_epochs if train_steps is None else train_steps/len(train_loader)*train_epochs
  if cycles is None:
    assert xargs.rea_epochs is not None
    cycles = xargs.rea_epochs / cycle_len # Just super large number because we are using epoch budget
    print(f"Converted cycles=None to cycles={cycles} since rea_epochs={xargs.rea_epochs} and each cycle has cycle_len={cycle_len}")
  # Initialize the population with random models.
  while len(population) < population_size:
    model = deepcopy(network)
    w_optimizer, w_scheduler, criterion = get_finetune_scheduler(xargs.scheduler, config, xargs, model, None)
    cur_arch = arch_sampler.random_topology_func()
    model.set_cal_mode("dynamic", cur_arch)

    metrics, sum_metrics = eval_archs_on_batch(xloader=train_loader, archs=[cur_arch], network = model, criterion=criterion, 
      train_steps=train_steps, epochs=train_epochs, same_batch=True, metric=metric, train_loader=train_loader, w_optimizer=w_optimizer, progress_bar=False)
    if xargs.rea_metric in ['loss', 'acc']:
      decision_metric, decision_lambda = metrics[0], lambda x: x[metric][0]
    elif xargs.rea_metric in ['sotl']:
      decision_metric, decision_lambda = sum_metrics["loss"], lambda x: x["sum"]["loss"]
    elif xargs.rea_metric in ['soacc']:
      decision_metric, decision_lambda = sum_metrics["acc"], lambda x: x["sum"]["acc"]
    model.metric = decision_metric
    model.arch = cur_arch
    ground_truth = summarize_results_by_dataset(cur_arch, api=api, iepoch=199, hp='200')
    history_stats = {"model":model, metric: metrics[0], "sum": sum_metrics, "arch": cur_arch, "ground_truth": ground_truth}

    # Append the info
    population.append(history_stats)
    history.append(history_stats)
    total_time += cycle_len

    top_n_perfs = sorted(history, key = decision_lambda, reverse=True) # Should start with the best and end with the worst

    # Reformatting history into top-N logging
    top_perfs = {}
    for top in top_ns:
      top_perf = {nth_top: top_n_perfs[min(nth_top, len(top_n_perfs)-1)]["ground_truth"]
        for nth_top in range(top)}
      top_perf = avg_nested_dict(top_perf)
      top_perfs["top"+str(top)] = top_perf

    cur_best_arch.append(top_n_perfs[0]["arch"].tostr())
    wandb.log({"ground_truth":top_perfs, "total_time": total_time})

  # Carry out evolution in cycles. Each cycle produces a model and removes another.
  for i in tqdm(range(round(cycles)), desc = "Cycling in REA"):
    # Sample randomly chosen models from the current population.
    if total_time >= xargs.rea_epochs:
      logger.log("Breaking REA early because the total budget was reached")
      break
    start_time, sample = time.time(), []
    while len(sample) < sample_size:
      # Inefficient, but written this way for clarity. In the case of neural
      # nets, the efficiency of this line is irrelevant because training neural
      # nets is the rate-determining step.
      candidate = random.choice(list(population))
      sample.append(candidate)

    # The parent is the best model in the sample.
    parent = max(sample, key=lambda i: i["model"].metric)

    # Create the child model and store it.
    child = deepcopy(network)
    w_optimizer, w_scheduler, criterion = get_finetune_scheduler(xargs.scheduler, config, xargs, child, None)

    cur_arch = mutate_arch(parent["model"].arch)
    child.arch = cur_arch
    child.set_cal_mode("dynamic", cur_arch)

    metrics, sum_metrics = eval_archs_on_batch(xloader=train_loader, archs=[cur_arch], network = child, criterion=criterion, train_steps=train_steps, epochs=train_epochs, same_batch=True, metric=metric, train_loader=train_loader, w_optimizer=w_optimizer)
    if xargs.rea_metric in ['loss', 'acc']:
      decision_metric, decision_lambda = metrics[0], lambda x: x[metric][0]
    elif xargs.rea_metric in ['sotl']:
      decision_metric, decision_lambda = sum_metrics["loss"], lambda x: x["sum"]["loss"]
    elif xargs.rea_metric in ['soacc']:
      decision_metric, decision_lambda = sum_metrics["acc"], lambda x: x["sum"]["acc"]
    child.metric = decision_metric
    child.arch = cur_arch
    ground_truth = summarize_results_by_dataset(cur_arch, api=api, iepoch=199, hp='200')
    history_stats = {"model":child, metric: metrics[0], "sum": sum_metrics, "arch": cur_arch, "ground_truth": ground_truth}


    # Append the info
    population.append(history_stats)
    history.append(history_stats)
    total_time += cycle_len

    top_n_perfs = sorted(history, key = decision_lambda, reverse=True) # Should start with the best and end with the worst

    # Reformatting history into top-N logging
    top_perfs = {}
    for top in top_ns:
      top_perf = {nth_top: top_n_perfs[nth_top]["ground_truth"]
        for nth_top in range(top)}
      top_perf = avg_nested_dict(top_perf)
      top_perfs["top"+str(top)] = top_perf

    cur_best_arch.append(top_n_perfs[0]["arch"].tostr())
    if i % 50 == 0:
      print(f"REA best perf at iter={i} is {top_n_perfs[0]['ground_truth']}")
    wandb.log({"ground_truth":top_perfs, "total_time": total_time})

    # Remove the oldest model.
    population.popleft()

  return history, cur_best_arch, total_time



def search_func_bare(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, algo, logger, args=None, epoch=None, smoke_test=False, 
  meta_learning=False, api=None, supernets_decomposition=None, arch_groups_quartiles=None, arch_groups_brackets: Dict=None, 
  all_archs=None, grad_metrics_percentiles=None, metrics_percs=None, percentiles=None, loss_threshold=None, replay_buffer = None, checkpoint_freq=3, val_loader=None, train_loader=None, meta_optimizer=None):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(track_std=True), AverageMeter(track_std=True), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(track_std=True), AverageMeter(track_std=True), AverageMeter()

  end = time.time()
  network.train()
  parsed_algo = algo.split("_")
  if args.search_space_paper == "nats-bench":
    if (len(parsed_algo) == 3 and ("perf" in algo or "size" in algo)): # Can be used with algo=random_size_highest etc. so that it gets parsed correctly
      arch_sampler = ArchSampler(api=api, model=network, mode=parsed_algo[1], prefer=parsed_algo[2], op_names=network._op_names, max_nodes = args.max_nodes, search_space = args.search_space_paper)
    else:
      arch_sampler = ArchSampler(api=api, model=network, mode="random", prefer="random", op_names=network._op_names, max_nodes = args.max_nodes, search_space = args.search_space_paper) # TODO mode=perf is a placeholder so that it loads the perf_all_dict, but then we do sample(mode=random) so it does not actually exploit the perf information
  else:
    arch_sampler = None
    
  arch_overview = {"cur_arch": None, "all_cur_archs": [], "all_archs": [], "top_archs_last_epoch": [], "train_loss": [], "train_acc": [], "val_acc": [], "val_loss": []}
  search_loader_iter = iter(xloader)
  if args.inner_steps is not None:
    inner_steps = args.inner_steps
  else:
    inner_steps = 1 # SPOS equivalent
  logger.log(f"Starting search with batch_size={len(next(iter(xloader)))}, len={len(xloader)}")
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in tqdm(enumerate(search_loader_iter), desc = "Iterating over SearchDataset", total = round(len(xloader)/(inner_steps if not args.inner_steps_same_batch else 1))): # Accumulate gradients over backward for sandwich rule
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, args)
    network.zero_grad()
    if smoke_test and step >= 3:
      break
    if step == 0:
      logger.log(f"New epoch (len={len(search_loader_iter)}) of arch; for debugging, those are the indexes of the first minibatch in epoch: {base_targets[0:10]}")
    scheduler.update(None, 1.0 * step / len(xloader))
    # measure data loading time
    data_time.update(time.time() - end)

    if (args.sandwich is None or args.sandwich == 1):
        outer_iters = 1
    else:
      outer_iters = args.sandwich
    if args.sandwich_mode in ["quartiles", "fairnas"]:
      sampled_archs = arch_sampler.sample(mode = args.sandwich_mode, subset = all_archs, candidate_num=args.sandwich) # Always samples 4 new archs but then we pick the one from the right quartile

    for outer_iter in range(outer_iters):
      # Update the weights
      # sampled_arch = sample_arch_and_set_mode(network, algo, arch_sampler)
      sampled_arch = None
      network.set_cal_mode("urs", None)
      
      if sampled_arch is not None:
        arch_overview["cur_arch"] = sampled_arch
        arch_overview["all_archs"].append(sampled_arch)
        arch_overview["all_cur_archs"].append(sampled_arch)

      fnetwork = network
      fnetwork.zero_grad()
      diffopt = w_optimizer
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
        if step in [0, 1] and inner_step < 3 and epoch % 5 == 0:
          logger.log(f"Base targets in the inner loop at inner_step={inner_step}, step={step}: {base_targets[0:10]}")
          # if algo.startswith("gdas"): # NOTE seems the forward pass doesnt explicitly change the genotype? The gumbels are always resampled in forward_gdas but it does not show up here
          #   logger.log(f"GDAS genotype at step={step}, inner_step={inner_step}, epoch={epoch}: {sampled_arch}")
        _, logits = fnetwork(base_inputs)
        base_loss = criterion(logits, base_targets) * (1 if args.sandwich is None else 1/args.sandwich)
        base_loss.backward()
        w_optimizer.step()
        network.zero_grad()
        base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
        base_losses.update(base_loss.item() / (1 if args.sandwich is None else 1/args.sandwich),  base_inputs.size(0))
        base_top1.update  (base_prec1.item(), base_inputs.size(0))
        base_top5.update  (base_prec5.item(), base_inputs.size(0))
        
      arch_loss = torch.tensor(10) # Placeholder in case it never gets updated here. It is not very useful in any case
      # Preprocess the hypergradients into desired form
      if algo == 'setn':
        network.set_cal_mode('joint')
      elif algo.startswith('gdas'):
        network.set_cal_mode('gdas', None)
      elif algo.startswith('darts'):
        network.set_cal_mode('joint', None)
      elif 'random' in algo:
        network.set_cal_mode('urs', None)
      elif algo != 'enas':
        raise ValueError('Invalid algo name : {:}'.format(algo))
      network.zero_grad()
      if algo == 'darts-v2' and not args.meta_algo:
        arch_loss, logits = backward_step_unrolled(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets, meta_learning=meta_learning)
        a_optimizer.step()
      elif (algo == 'random' or algo == 'enas' or 'random' in algo ) and not args.meta_algo:
        if algo == "random" and args.merge_train_val_supernet:
          arch_loss = torch.tensor(10) # Makes it slower and does not return anything useful anyways
        else:
          arch_loss = torch.tensor(10)
          # with torch.no_grad():
          #   _, logits = network(arch_inputs)
          #   arch_loss = criterion(logits, arch_targets)
      else:
        # The Darts-V1/FOMAML/GDAS/who knows what else branch
        network.zero_grad()
        _, logits = network(arch_inputs)
        arch_loss = criterion(logits, arch_targets)
        arch_loss.backward()
        a_optimizer.step()
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
      arch_overview["val_acc"].append(arch_prec1)
      arch_overview["val_loss"].append(arch_loss.item())
      arch_overview["all_cur_archs"] = [] #Cleanup
  network.zero_grad()
  # measure elapsed time
  batch_time.update(time.time() - end)
  end = time.time()
  
  if step % print_freq == 0 or step + 1 == len(xloader):
    Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
    Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
    Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
    Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
    logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
    if step == print_freq:
      logger.log(network.show_alphas())

  eigenvalues=None
  search_metric_stds,supernet_train_stats, supernet_train_stats_by_arch = {}, {}, {}
  search_metric_stds = {"train_loss.std": base_losses.std, "train_loss_arch.std": base_losses.std, "train_acc.std": base_top1.std, "train_acc_arch.std": arch_top1.std}
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg, supernet_train_stats, supernet_train_stats_by_arch, arch_overview, search_metric_stds, eigenvalues


def train_epoch(train_loader, network, w_optimizer, criterion, algo, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  loss, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  if algo.startswith('setn'):
    sampled_arch = network.dync_genotype(True)
    network.set_cal_mode('dynamic', sampled_arch)
  elif algo.startswith('gdas'):
    network.set_cal_mode('gdas', None)
    sampled_arch = network.genotype
  elif algo.startswith('darts'):
    network.set_cal_mode('joint', None)
    sampled_arch = network.genotype
  
  elif "random" in algo: # TODO REMOVE SOON
    network.set_cal_mode('urs')
  start = time.time()
  for step, (inputs, targets) in tqdm(enumerate(train_loader), desc = "Iterating over batches while training weights only", total = len(train_loader)):
    targets = targets.cuda(non_blocking=True)
    _, logits = network(inputs.cuda(non_blocking=True))
    train_loss = criterion(logits, targets)
    train_loss.backward()
    w_optimizer.step()
    network.zero_grad()
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    loss.update(train_loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))
  end = time.time()
  logger.log(f"Trained epoch in {end-start} time, avg loss = {loss.avg}, avg acc = {top1.avg}")
  return loss.avg, top1.avg, top5.avg


def evenify_training(network2, train_loader, criterion, w_optimizer2, logger, arch_idx, epoch_eqs, sampled_arch):
    # Train each architecture until they all reach the same amount of training as a preprocessing step before recording the training statistics for correlations
    cur_epoch, target_loss = epoch_eqs[sampled_arch.tostr()]["epoch"], epoch_eqs[sampled_arch.tostr()]["val"]
    max_epoch_attained = max([x["val"] for x in epoch_eqs.values()])
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
    logger.log(f"Trained arch_idx for {iter_count} iterations to make it match up to {max_epoch_attained}")
    
def exact_hessian(network, val_loader, criterion, xloader, epoch, logger, args):
  labels = []
  for i in range(network._max_nodes):
    for n in network._op_names:
      labels.append(n + "_" + str(i))

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
                                                             power_iter_steps=50, arch_only=True, full_dataset=True)
  val_dom_eigenvalue = val_eigenvals[0]
  try:
    if hasattr(args, "merge_train_val_supernet") and not args.merge_train_val_supernet:
      train_eigenvals, train_eigenvecs = compute_hessian_eigenthings(network, val_loader, criterion, 1, mode="power_iter", 
                                                                    power_iter_steps=50, arch_only=True, full_dataset=True)
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
  network.zero_grad()
  return eigenvalues

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


def backward_step_unrolled_darts(network, criterion, base_inputs, base_targets, w_optimizer, arch_inputs, arch_targets):
  # _compute_unrolled_model

  _, logits = network(base_inputs)
  loss = criterion(logits, base_targets)
  start=time.time()
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
  # print(f"Time of momentum whatever: {time.time()-start}")
  start=time.time()
  unrolled_model = deepcopy(network)
  model_dict  = unrolled_model.state_dict()
  new_params, offset = {}, 0
  start2=time.time()
  for k, v in network.named_parameters():
    if 'arch' in k or 'alpha' in k: continue
    v_length = np.prod(v.size())
    model_dict[k] = params[offset: offset+v_length].view(v.size())
    offset += v_length
    
  # print(f"Loading shit subroutine : {time.time()-start2}")
  # model_dict.update(new_params)
  # unrolled_model.load_state_dict(model_dict)
  # print(f"Loading shit {time.time()-start}")

  start=time.time()
  unrolled_model.zero_grad()
  _, unrolled_logits = unrolled_model(arch_inputs)
  unrolled_loss = criterion(unrolled_logits, arch_targets)
  unrolled_loss.backward()
  # print(f"Model forward: {time.time()-start}")

  dalpha = [p.grad for p in unrolled_model.arch_parameters]
  vector = [v.grad.data for v in unrolled_model.weights]
  start=time.time()
  implicit_grads = _hessian_vector_product(vector, network, criterion, base_inputs, base_targets)
  # print(f"time of hvp: {time.time()-start}")
  
  for g, ig in zip(dalpha, implicit_grads):
    # dalpha.data.sub_(LR, implicit_grads.data)
    g.data.sub_(LR, ig.data)

  for p, da in zip(network.arch_parameters, dalpha):
    if p.grad is None:
      p.grad = deepcopy( da )
    else:
      p.data.copy_( da.data )
  return unrolled_loss.detach(), unrolled_logits.detach()

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

def update_supernets_decomposition(supernets_decomposition, arch_groups_quartiles, losses_percs, grad_metrics_percentiles, base_loss, data_step, epoch, xloader, sampled_arch,
                                   fnetwork):
    # TODO need to fix the logging here I think. The normal logging is much better now
    cur_quartile = arch_groups_quartiles[sampled_arch.tostr()]
    with torch.no_grad():
        dw = [p.grad.detach().to('cpu') if p.grad is not None else torch.zeros_like(p).to('cpu') for p in
              fnetwork.parameters()]
        cur_supernet = supernets_decomposition[cur_quartile]
        for decomp_w, g in zip(cur_supernet.parameters(), dw):
            if decomp_w.grad is not None:
                decomp_w.grad.copy_(g)
            else:
                decomp_w.grad = g
        analyze_grads(cur_supernet, grad_metrics_percentiles["perc" + str(cur_quartile)]["supernet"],
                      true_step=data_step + epoch * len(xloader), total_steps=data_step + epoch * len(xloader))

    if type(arch_groups_quartiles) is dict:
        for quartile in arch_groups_quartiles.keys():
            if quartile == cur_quartile:
                losses_percs["perc" + str(quartile)].update(base_loss.item())  # TODO this doesnt make any sense
                
def bracket_tracking_setup(arch_groups_brackets, brackets_cond, arch_sampler):
  all_brackets = set(arch_groups_brackets.values()) if brackets_cond else set()
  supernet_train_stats = {"train_loss":{"sup"+str(percentile): [] for percentile in all_brackets}, 
    "val_loss": {"sup"+str(percentile): [] for percentile in all_brackets},
    "val_acc": {"sup"+str(percentile): [] for percentile in all_brackets},
    "train_acc": {"sup"+str(percentile): [] for percentile in all_brackets}}
  supernet_train_stats_by_arch = {arch: {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []} for arch in (arch_sampler.archs if arch_sampler is not None else {})}
  supernet_train_stats_avgmeters = {}
  for k in list(supernet_train_stats.keys()):
    supernet_train_stats[k+str("AVG")] = {"sup"+str(percentile): [] for percentile in all_brackets}
    supernet_train_stats_avgmeters[k+str("AVG")] = {"sup"+str(percentile): AverageMeter() for percentile in all_brackets}
  return all_brackets, supernet_train_stats, supernet_train_stats_by_arch, supernet_train_stats_avgmeters


def update_running(running, valid_loss=None, valid_acc = None, valid_acc_top5=None, loss=None, train_acc_top1=None, 
                   train_acc_top5=None, sogn=None, sogn_norm=None, total_train_loss_for_sotl_aug=None):
  if valid_loss is not None:
    running["sovl"] -= valid_loss
  if valid_acc is not None:
    running["sovalacc"] += valid_acc
  # if valid_acc_top5 is not None:
  #   running["sovalacc_top5"] += valid_acc_top5
  # if train_acc_top5 is not None:
  #   running["sotrainacc_top5"] += train_acc_top5
  if loss is not None:
    running["sotl"] -= loss # Need to have negative loss so that the ordering is consistent with val acc
  if train_acc_top1 is not None:
    running["sotrainacc"] += train_acc_top1

  if sogn is not None:
    # running["sogn"] += grad_metrics["train"]["sogn"]
    running["sogn"] += sogn
  if sogn_norm is not None:
    # running["sogn_norm"] += grad_metrics["train"]["grad_normalized"]
    running["sogn_norm"] += sogn_norm
  if total_train_loss_for_sotl_aug is not None:
    # running["sotl_aug"] = running["sotl"] + total_metrics_dict["total_train_loss"]
    running["sotl_aug"] = running["sotl"] + total_train_loss_for_sotl_aug
  if valid_loss is not None and loss is not None:
    running["sobothl"] -= (valid_loss + loss)
  return running

def update_base_metrics(metrics, running, metrics_keys=None, grad_metrics=None, drop_fancy=False, grads_analysis=None, 
                        valid_acc=None, train_acc=None, loss=None, valid_loss=None, arch_str=None, epoch_idx = None):
  if metrics_keys is None:
    metrics_keys = metrics.keys()
  for k in running.keys():
    metrics[k][arch_str][epoch_idx].append(running[k])
  metrics["val_acc"][arch_str][epoch_idx].append(valid_acc)
  metrics["train_acc"][arch_str][epoch_idx].append(train_acc)
  metrics["train_loss"][arch_str][epoch_idx].append(-loss)
  metrics["val_loss"][arch_str][epoch_idx].append(-valid_loss)
  metrics["gap_loss"][arch_str][epoch_idx].append(-valid_loss + (loss - valid_loss))
  # if arch_str is not None and epoch_idx is not None:
  #   if len(metrics["train_loss"][arch_str][epoch_idx]) >= 3:
  #     loss_normalizer = sum(metrics["train_loss"][arch_str][epoch_idx][-3:])/3
  #   elif epoch_idx >= 1:
  #     loss_normalizer = sum(metrics["train_loss"][arch_str][epoch_idx-1][-3:])/3
  #   else:
  #     loss_normalizer = 1
  #   metrics["train_loss_pct"][arch_str][epoch_idx].append(1-loss/loss_normalizer)
  data_types = ["train"] if not grads_analysis else ["train", "val", "total_train", "total_val"]
  grad_log_keys = ["gn", "gnL1", "sogn", "sognL1", "grad_normalized", "grad_accum", "grad_accum_singleE", "grad_accum_decay", "grad_mean_accum", "grad_mean_sign", "grad_var_accum", "grad_var_decay_accum"]

  if not drop_fancy and grad_metrics is not None:
    for data_type in data_types:
      for log_key in grad_log_keys:
        val = grad_metrics[data_type][log_key]
        metrics[data_type+"_"+log_key][arch_str][epoch_idx].append(grad_metrics[data_type][log_key])
  return metrics

def load_my_state_dict(model, state_dict):
  own_state = model.state_dict()
  for name, param in state_dict.items():
      if name not in own_state or 'classifier' in name:
            continue
      if isinstance(param, torch.nn.Parameter):
          # backwards compatibility for serialized parameters
          param = param.data
      own_state[name].copy_(param)
      
def resolve_higher_conds(xargs):
  use_higher_cond = xargs.meta_algo and xargs.meta_algo not in ['reptile', 'metaprox']
  if xargs.meta_algo is not None and 'darts' in xargs.meta_algo and xargs.higher_method == "joint" and (xargs.sandwich is None or xargs.sandwich == 1): # Special case for single-level DARTS training
    print("Set use_higher_cond to False because using single-level DARTS most likely")
    use_higher_cond = False 
  
  diffopt_higher_grads_cond = True if (xargs.meta_algo not in ['reptile', 'metaprox', 'reptile_higher'] and xargs.higher_order != "first") else False
  monkeypatch_higher_grads_cond = True if (xargs.meta_algo not in ['reptile', 'metaprox', 'reptile_higher'] and (xargs.higher_order != "first" or xargs.higher_method == "val")) else False
  first_order_grad_for_free_cond = xargs.higher_order == "first" and xargs.higher_method in ["sotl", "sotl_v2"]
  first_order_grad_concurrently_cond = xargs.higher_order == "first" and xargs.higher_method.startswith("val")
  second_order_grad_optimization_cond = xargs.higher_order == "second" and xargs.higher_method in ["sotl", "sotl_v2"]
  print(f"Resolved higher conds as use_higher_cond={use_higher_cond}, diffopt_higher_grads_cond={diffopt_higher_grads_cond}, monkeypatch_higher_grads_cond={monkeypatch_higher_grads_cond}, first_order_grad_for_free_cond={first_order_grad_for_free_cond}, first_order_grad_concurrently_cond={first_order_grad_concurrently_cond}, second_order_grad_optimization_cond={second_order_grad_optimization_cond}")
  return use_higher_cond, diffopt_higher_grads_cond, monkeypatch_higher_grads_cond, first_order_grad_for_free_cond, first_order_grad_concurrently_cond, second_order_grad_optimization_cond


def init_search_from_checkpoint(search_model, logger, xargs):
  # The supernet init path can have form like '1,2,3' or 'darts_1,darts_2,darts_3' or 'cifar10_random_1, cifar10_random_2, cifar10_random_3'
  
  split_path = xargs.supernet_init_path.split(",")
  whole_path = split_path[xargs.rand_seed % len(split_path)]
  logger.log(f"Picked {xargs.rand_seed % len(split_path)}-th seed from {xargs.supernet_init_path}")
  if os.path.exists(xargs.supernet_init_path):
    pass
  else:
    try:
      dataset, algo = "cifar10", "random" # Defaults
      parsed_init_path = whole_path.split("_") # Should be algo followed by seed number, eg. darts_1 or random_30 or cifar100_random_50
      logger.log(f"Parsed init path into {parsed_init_path}")
      if len(parsed_init_path) == 2:
        seed_num = int(parsed_init_path[1])
        seed_algo = parsed_init_path[0]
      if len(parsed_init_path) == 3:
        seed_num = int(parsed_init_path[2])
        seed_algo = parsed_init_path[1]
        dataset = parsed_init_path[0]
      whole_path = f'./output/search-tss/{dataset}/{seed_algo}-affine0_BN0-None/checkpoint/seed-{seed_num}-basic.pth'
    except Exception as e:
      logger.log(f"Supernet init path does not seem to be formatted as seed number - it is {xargs.supernet_init_path}, error was {e}")
  
  logger.log(f'Was given supernet checkpoint to use as initialization at {xargs.supernet_init_path}, decoded into {whole_path} and loaded its weights into search model')
  checkpoint = torch.load(whole_path)
  # The remaining things that are usually contained in a checkpoint are restarted to empty a bit further down
  
  search_model.load_state_dict(checkpoint['search_model'], strict=False)
  # load_my_state_dict(model, checkpoint["search_model"])
  
def init_supernets_decomposition(xargs, logger, checkpoint, network):
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
  
  return percentiles, supernets_decomposition, arch_groups_quartiles, archs_subset, grad_metrics_percs, metrics_factory, metrics_percs

def scheduler_step(w_scheduler2, epoch_idx, batch_idx, train_loader, steps_per_epoch, scheduler_type):
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
    
def count_ops(arch):
  ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
  arch_str = str(arch)
  counts = {op: arch_str.count(op) for op in ops}
  return counts

def grad_drop(params, p=0.0, arch_param_count=None, p_method=None):
  if p == 0:
    pass
  else:
    # NB201 param avg: 0.3985MB
    for param in params:
      if param.requires_grad and param.grad is not None:
        if p_method is None:
          torch.nn.functional.dropout(param.grad, p,  training = True, inplace = True)
        elif p_method == "adaptive":
          p = None
        else:
          raise NotImplementedError
        
        
def search_func_old(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, algo, logger):
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




def train_real(xargs, use_higher_cond, network, fnetwork, criterion, before_rollout_state, logger, all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets, w_optimizer, epoch, data_step, outer_iter, outer_iters):

  if use_higher_cond and xargs.higher_loop == "bilevel" and xargs.higher_params == "arch" and xargs.sandwich_computation == "serial" and xargs.meta_algo not in ["reptile", "metaprox"]:
    if xargs.refresh_arch_oneshot in ["always", "train_real"]: network.refresh_arch_oneshot = True
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
    if xargs.refresh_arch_oneshot in ["train_real"]: network.refresh_arch_oneshot = True

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
      
      
      
      

def get_best_arch_old(train_loader, valid_loader, network, n_samples, algo, logger, 
  additional_training=True, api=None, style:str='sotl', w_optimizer=None, w_scheduler=None, 
  config: Dict=None, epochs:int=1, steps_per_epoch:int=100, 
  val_loss_freq:int=1, overwrite_additional_training:bool=False, 
  scheduler_type:str=None, xargs=None):
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
      decision_metrics = []    
      start_arch_idx = 0


    train_start_time = time.time()

    train_stats = [[] for _ in range(epochs*steps_per_epoch+1)]

    for arch_idx, sampled_arch in tqdm(enumerate(archs[start_arch_idx:], start_arch_idx), desc="Iterating over sampled architectures", total = n_samples-start_arch_idx):
      network2 = deepcopy(network)
      network2.set_cal_mode('dynamic', sampled_arch)

      if xargs.lr is not None and scheduler_type is None:
        scheduler_type = "constant"

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

      elif xargs.lr is not None and scheduler_type == 'constant':
        config = config._replace(scheduler='constant', constant_lr=xargs.lr)
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

      # q = mp.Queue()
      # # This reporting process is necessary due to WANDB technical difficulties. It is used to continuously report train stats from a separate process
      # # Otherwise, when a Run is intiated from a Sweep, it is not necessary to log the results to separate training runs. But that it is what we want for the individual arch stats
      # p=mp.Process(target=train_stats_reporter, kwargs=dict(queue=q, config=vars(xargs),
      #     sweep_group=f"Search_Cell_{algo}_arch", sweep_run_name=wandb.run.name or wandb.run.id or "unknown", arch=sampled_arch.tostr()))
      # p.start()

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
            elif scheduler_type == "cos_reinit":
              w_scheduler2.update(epoch_idx, 0.0)
            elif scheduler_type in ['cos_fast', 'cos_warmup']:
              w_scheduler2.update(epoch_idx , batch_idx/min(len(train_loader), steps_per_epoch))
            else:
              w_scheduler2.update(None, 1.0 * batch_idx / len(train_loader))


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
          # q.put(batch_train_stats)
          train_stats[epoch_idx*steps_per_epoch+batch_idx].append(batch_train_stats)

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

      # q.put("SENTINEL") # This lets the Reporter process know it should quit

            
    train_total_time = time.time()-train_start_time
    print(f"Train total time: {train_total_time}")

    wandb.run.summary["train_total_time"] = train_total_time

    original_metrics = deepcopy(metrics)

    metrics_FD = {k+"FD": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=1).get_time_series(chunked=True, mode="fd") for arch in archs} for k,v in metrics.items() if k in ['val', 'train_losses', 'val_losses']}
    metrics.update(metrics_FD)
    if epochs > 1:
      interim = {} # We need an extra dict to avoid changing the dict's keys during iteration for the R metrics
      for key in metrics.keys():
        if key in ["train_losses", "train_lossesFD", "val_losses", "val"]:
          interim[key+"R"] = {}
          for arch in archs:
            arr = []
            for epoch_idx in range(len(metrics[key][arch.tostr()])):
              epoch_arr = []
              for batch_metric in metrics[key][arch.tostr()][epoch_idx]:
                if key in ["train_losses", "train_lossesFD", "val_losses"]:
                  sign = -1
                else:
                  sign = 1
                epoch_arr.append(sign*batch_metric if epoch_idx == 0 else -1*sign*batch_metric)
              arr.append(epoch_arr)
            interim[key+"R"][arch.tostr()] = SumOfWhatever(measurements=arr, e=epochs+1, mode='last').get_time_series(chunked=True)
            # interim[key+"R"][arch.tostr()] = SumOfWhatever(measurements=[[[batch_metric if epoch_idx == 0 else -batch_metric for batch_metric in batch_metrics] for batch_metrics in metrics[key][arch.tostr()][epoch_idx]]] for epoch_idx in range(len(metrics[key][arch.tostr()])), e=epochs+1).get_time_series(chunked=True)
      
      # print(interim)
      # print(metrics["train_lossesFD"])
      # print(metrics["train_losses"])
      metrics.update(interim)

      metrics_E1 = {k+"E1": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=1).get_time_series(chunked=True) for arch in archs} for k,v in metrics.items()}
      metrics.update(metrics_E1)

    else:
      # We only calculate Sum-of-FD metrics in this case
      metrics_E1 = {k+"E1": {arch.tostr():SumOfWhatever(measurements=metrics[k][arch.tostr()], e=1).get_time_series(chunked=True) for arch in archs} for k,v in metrics.items() if "FD" in k}
      metrics.update(metrics_E1)
    for key in metrics_FD.keys(): # Remove the pure FD metrics because they are useless anyways
      metrics.pop(key, None)


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
    corr_metrics_path = save_checkpoint({"metrics":original_metrics, "corrs": corrs, 
      "archs":archs, "start_arch_idx":arch_idx+1, "config":vars(xargs), "decision_metrics":decision_metrics},
      logger.path('corr_metrics'), logger)
    try:
      wandb.save(str(corr_metrics_path.absolute()))
    except:
      print("Upload to WANDB failed")

  best_idx = np.argmax(decision_metrics)
  try:
    best_arch, best_valid_acc = archs[best_idx], decision_metrics[best_idx]
  except:
    logger.log("Failed to get best arch via decision_metrics")
    logger.log(f"Decision metrics: {decision_metrics}")
    logger.log(f"Best idx: {best_idx}, length of archs: {len(archs)}")
    best_arch,best_valid_acc = archs[0], decision_metrics[0]
  return best_arch, best_valid_acc
