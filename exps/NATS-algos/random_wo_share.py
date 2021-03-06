##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##############################################################################
# Random Search for Hyper-Parameter Optimization, JMLR 2012 ##################
##############################################################################
# python ./exps/NATS-algos/random_wo_share.py --dataset cifar10 --search_space tss --rand_seed 777 --true_ranking_hp 200 --hp 12 --time_budget 200000
# python ./exps/NATS-algos/random_wo_share.py --dataset cifar100 --search_space tss
# python ./exps/NATS-algos/random_wo_share.py --dataset ImageNet16-120 --search_space tss
##############################################################################
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
from regularized_ea import random_topology_func, random_size_func
from utils.sotl_utils import (simulate_train_eval_sotl, query_all_results_by_arch, wandb_auth, summarize_results_by_dataset, 
  simulate_train_eval_sotl_whole_history, get_true_rankings, calc_corrs_after_dfs)
import wandb


def main(xargs, api):
  torch.set_num_threads(4)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  logger.log('{:} use api : {:}'.format(time_string(), api))
  api.reset_time()

  search_space = get_search_spaces(xargs.search_space, 'nats-bench')
  if xargs.search_space == 'tss':
    random_arch = random_topology_func(search_space)
  else:
    random_arch = random_size_func(search_space)

  best_arch, best_acc, total_time_cost, history = None, -1, [], []
  current_best_index = []

  archs = []
  metrics_per_arch = {}
  cand_total = 0
  while (len(total_time_cost) == 0 or total_time_cost[-1] < xargs.time_budget or len(total_time_cost) > 16000) and cand_total <= args.eval_candidate_num:
    arch = random_arch()
    cand_total += 1

    archs.append(arch)

    accuracy, _, _, total_cost = simulate_train_eval_sotl(api=api, arch=arch, dataset=xargs.dataset, 
      iepoch=xargs.epoch, hp=xargs.hp, metric=xargs.metric, e=xargs.e)

    metrics, _ = simulate_train_eval_sotl_whole_history(api=api, arch=arch, 
      dataset=xargs.dataset, hp=xargs.hp, metric=xargs.metric, e=xargs.e)
    
    metrics_per_arch[arch.tostr()] = [metrics] # The assigned item should ba sequence since the API was designed for supernet training which works on Epoch->Minibatch indexing


    total_time_cost.append(total_cost)
    history.append(arch)
    if best_arch is None or best_acc < accuracy:
      best_acc, best_arch = accuracy, arch
    logger.log('[{:03d}] : {:} : accuracy = {:.2f}%'.format(len(history), arch, accuracy))
    current_best_index.append(api.query_index_by_arch(best_arch))
  logger.log('{:} best arch is {:}, accuracy = {:.2f}%, visit {:} archs with {:.1f} s.'.format(time_string(), best_arch, best_acc, len(history), total_time_cost[-1]))
  
  info = api.query_info_str_by_arch(best_arch, '200' if xargs.search_space == 'tss' else '90')

  abridged_results = query_all_results_by_arch(best_arch, api, iepoch=199, hp='200')

  true_rankings, final_accs = get_true_rankings(archs, api, hp=xargs.true_ranking_hp)
  corrs, to_log = calc_corrs_after_dfs(epochs=1, xloader=[None]*(200 if xargs.hp == '200' else 12), steps_per_epoch=None, metrics_depth_dim=metrics_per_arch, 
    final_accs = final_accs, archs=archs, true_rankings = true_rankings, prefix=xargs.metric, api=api, wandb_log=False)
  for epoch_idx in range(len(to_log)):
    for batch_idx in range(len(to_log[0])):
      wandb.log(to_log[epoch_idx][batch_idx])
  logger.log('{:}'.format(info))
  logger.log('-'*100)
  logger.close()
  return logger.log_dir, current_best_index, total_time_cost, abridged_results


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Random NAS")
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120', 'cifar10-valid'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--search_space',       type=str,   choices=['tss', 'sss'], default='tss', help='Choose the search space.')

  parser.add_argument('--time_budget',        type=int,   default=20000, help='The total time cost budge for searching (in seconds).')
  parser.add_argument('--loops_if_rand',      type=int,   default=500,   help='The total runs for evaluation.')
  # log
  parser.add_argument('--save_dir',           type=str,   default='./output/search', help='Folder to save checkpoints and log.')
  parser.add_argument('--rand_seed',          type=int,   default=-1,    help='manual seed')
  parser.add_argument('--metric', type=str, default='valid-accuracy', help='valid-accuracy/train-loss/valid-loss')
  parser.add_argument('--epoch', type=int, default=11, help='Default should be hp-1')
  parser.add_argument('--hp', type=str, default='12', help='12 or 200')
  parser.add_argument('--e', type=int, default=1, help='SOTL-E')
  parser.add_argument('--true_ranking_hp', type=str, default='200', help='12 or 200')
  parser.add_argument('--eval_candidate_num', type=int, default=200, help='12 or 200')


  args = parser.parse_args()
  
  if 'TORCH_HOME' not in os.environ:
    if os.path.exists('/storage/.torch/'):
      os.environ["TORCH_HOME"] = '/storage/.torch/'

  api = create(None, args.search_space, fast_mode=True, verbose=False)
  print(f" TEST: {api.get_more_info(35, 'cifar100')}")
  args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space),
                               '{:}-T{:}'.format(args.dataset, args.time_budget), 'RANDOM')
  print('save-dir : {:}'.format(args.save_dir))

  wandb_auth()

  if args.rand_seed < 0:


    save_dir, all_info = None, collections.OrderedDict()
    results_summary = []
    for i in range(args.loops_if_rand):
      run =wandb.init(project="NAS", group="RS_no_share_individual", reinit=True)
      wandb.config.update(args)
      if i % 10 == 0:
        api = create(None, args.search_space, fast_mode=True, verbose=False)
      print ('{:} : {:03d}/{:03d}'.format(time_string(), i, args.loops_if_rand))
      args.rand_seed = random.randint(1, 100000)
      save_dir, all_archs, all_total_times, abridged_results = main(args, api)
      results_summary.append(abridged_results)
      all_info[i] = {'all_archs': all_archs,
                     'all_total_times': all_total_times}

      run.finish()

    interim = summarize_results_by_dataset(results_summary=results_summary, separate_mean_std=True)

    print(interim)
    save_path = save_dir / 'results.pth'
    print('save into {:}'.format(save_path))
    torch.save(all_info, save_path)

    wandb.init(project="NAS", group="RS_no_share_group", reinit=True)

    wandb.log(interim)
  else:
    wandb.init(project="NAS", group="RS_no_share_group", reinit=True)

    main(args, api)
