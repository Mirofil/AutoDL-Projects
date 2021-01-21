##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###################################################################
# BOHB: Robust and Efficient Hyperparameter Optimization at Scale #
# required to install hpbandster ##################################
# pip install hpbandster         ##################################
###################################################################
# OMP_NUM_THREADS=4 python exps/NATS-algos/bohb.py --search_space tss --dataset cifar10 --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3 --rand_seed 1 --loops_if_rand 1
# OMP_NUM_THREADS=4 python exps/NATS-algos/bohb.py --search_space sss --dataset cifar10 --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3 --rand_seed 1
###################################################################
import os, sys, time, random, argparse, collections
from copy import deepcopy
from pathlib import Path
import torch
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger
from log_utils    import AverageMeter, time_string, convert_secs2time
from nats_bench   import create
from models       import CellStructure, get_search_spaces
# BOHB: Robust and Efficient Hyperparameter Optimization at Scale, ICML 2018
import ConfigSpace
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from utils.sotl_utils import simulate_train_eval_sotl, query_all_results_by_arch


def get_topology_config_space(search_space, max_nodes=4):
  cs = ConfigSpace.ConfigurationSpace()
  #edge2index   = {}
  for i in range(1, max_nodes):
    for j in range(i):
      node_str = '{:}<-{:}'.format(i, j)
      cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space))
  return cs


def get_size_config_space(search_space):
  cs = ConfigSpace.ConfigurationSpace()
  for ilayer in range(search_space['numbers']):
    node_str = 'layer-{:}'.format(ilayer)
    cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(node_str, search_space['candidates']))
  return cs


def config2topology_func(max_nodes=4):
  def config2structure(config):
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name = config[node_str]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
  return config2structure


def config2size_func(search_space):
  def config2structure(config):
    channels = []
    for ilayer in range(search_space['numbers']):
      node_str = 'layer-{:}'.format(ilayer)
      channels.append(str(config[node_str]))
    return ':'.join(channels)
  return config2structure


class MyWorker(Worker):

  def __init__(self, *args, convert_func=None, dataset=None, api=None, hp='12', metric='valid-accuracy', e=1, **kwargs):
    super().__init__(*args, **kwargs)
    self.convert_func   = convert_func
    self._dataset       = dataset
    self._api           = api
    self.e = e
    self.hp = hp
    self.metric = metric 
    self.total_times    = []
    self.trajectory     = []

  def compute(self, config, budget, **kwargs):
    arch  = self.convert_func( config )
    accuracy, latency, time_cost, total_time = simulate_train_eval_sotl(self._api, arch, self._dataset, iepoch=int(budget)-1, hp=self.hp, e= self.e, metric = self.metric)
    self.trajectory.append((accuracy, arch))
    self.total_times.append(total_time)
    return ({'loss': 100 - accuracy,
             'info': self._api.query_index_by_arch(arch)})


def main(xargs, api):
  torch.set_num_threads(4)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  logger.log('{:} use api : {:}'.format(time_string(), api))
  api.reset_time()
  search_space = get_search_spaces(xargs.search_space, 'nats-bench')
  if xargs.search_space == 'tss':
    cs = get_topology_config_space(search_space)
    config2structure = config2topology_func()
  else:
    cs = get_size_config_space(search_space)
    config2structure = config2size_func(search_space)
  
  hb_run_id = '0'

  NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
  ns_host, ns_port = NS.start()
  num_workers = 1

  workers = []
  for i in range(num_workers):
    w = MyWorker(nameserver=ns_host, nameserver_port=ns_port, convert_func=config2structure, dataset=xargs.dataset, api=api, run_id=hb_run_id, hp=xargs.hp, metric=xargs.metric, e=xargs.e, id=i)
    w.run(background=True)
    workers.append(w)

  start_time = time.time()
  bohb = BOHB(configspace=cs, run_id=hb_run_id,
      eta=3, min_budget=1, max_budget=12,
      nameserver=ns_host,
      nameserver_port=ns_port,
      num_samples=xargs.num_samples,
      random_fraction=xargs.random_fraction, bandwidth_factor=xargs.bandwidth_factor,
      ping_interval=10, min_bandwidth=xargs.min_bandwidth)
  
  results = bohb.run(xargs.n_iters, min_n_workers=num_workers)

  bohb.shutdown(shutdown_workers=True)
  NS.shutdown()

  # print('There are {:} runs.'.format(len(results.get_all_runs())))
  # workers[0].total_times
  # workers[0].trajectory
  current_best_index = []
  for idx in range(len(workers[0].trajectory)):
    trajectory = workers[0].trajectory[:idx+1]
    arch = max(trajectory, key=lambda x: x[0])[1]
    current_best_index.append(api.query_index_by_arch(arch))
  
  best_arch = max(workers[0].trajectory, key=lambda x: x[0])[1]
  logger.log('Best found configuration: {:} within {:.3f} s'.format(best_arch, workers[0].total_times[-1]))
  info = api.query_info_str_by_arch(best_arch, '200' if xargs.search_space == 'tss' else '90')

  abridged_results = query_all_results_by_arch(best_arch, api, iepoch=199, hp='200')
  logger.log('{:}'.format(info))
  logger.log('-'*100)
  logger.close()

  return logger.log_dir, current_best_index, workers[0].total_times, abridged_results


if __name__ == '__main__':
  parser = argparse.ArgumentParser("BOHB: Robust and Efficient Hyperparameter Optimization at Scale")
  parser.add_argument('--dataset',            type=str,  choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # general arg
  parser.add_argument('--search_space',       type=str,  choices=['tss', 'sss'], help='Choose the search space.')
  parser.add_argument('--time_budget',        type=int,  default=20000, help='The total time cost budge for searching (in seconds).')
  parser.add_argument('--loops_if_rand',      type=int,  default=500, help='The total runs for evaluation.')
  # BOHB
  parser.add_argument('--strategy', default="sampling",  type=str, nargs='?', help='optimization strategy for the acquisition function')
  parser.add_argument('--min_bandwidth',    default=.3,  type=float, nargs='?', help='minimum bandwidth for KDE')
  parser.add_argument('--num_samples',      default=64,  type=int, nargs='?', help='number of samples for the acquisition function')
  parser.add_argument('--random_fraction',  default=.33, type=float, nargs='?', help='fraction of random configurations')
  parser.add_argument('--bandwidth_factor', default=3,   type=int, nargs='?', help='factor multiplied to the bandwidth')
  parser.add_argument('--n_iters',          default=300, type=int, nargs='?', help='number of iterations for optimization method')
  # log
  parser.add_argument('--save_dir',           type=str,  default='./output/search', help='Folder to save checkpoints and log.')
  parser.add_argument('--rand_seed',          type=int,  default=-1, help='manual seed')
  parser.add_argument('--metric', type=str, default='valid-accuracy', help='validation-accuracy/train-loss/valid-loss')
  parser.add_argument('--hp', type=str, default='12', help='12 or 200')
  parser.add_argument('--e', type=int, default=1, help='SOTL-E')
  args = parser.parse_args()
  
  api = create(None, args.search_space, fast_mode=True, verbose=False)

  args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space),
                               '{:}-T{:}'.format(args.dataset, args.time_budget), 'BOHB')
  print('save-dir : {:}'.format(args.save_dir))
  if 'TORCH_HOME' not in os.environ:
    if os.path.exists('/notebooks/storage/.torch/'):
      os.environ["TORCH_HOME"] = '/notebooks/storage/.torch/'
  if args.rand_seed < 0:
    save_dir, all_info = None, collections.OrderedDict()
    results_summary = []

    for i in range(args.loops_if_rand):
      if i % 10 == 0:
        api = create(None, args.search_space, fast_mode=True, verbose=False)
      print ('{:} : {:03d}/{:03d}'.format(time_string(), i, args.loops_if_rand))
      args.rand_seed = random.randint(1, 100000)
      save_dir, all_archs, all_total_times, abridged_results = main(args, api)
      results_summary.append(abridged_results)
      all_info[i] = {'all_archs': all_archs,
                     'all_total_times': all_total_times}
                     
    interim = {}
    for dataset in results_summary[0].keys():
      interim[dataset]= {"mean":round(sum([result[dataset] for result in results_summary])/len(results_summary), 2),
        "std": round(np.std(np.array([result[dataset] for result in results_summary])), 2)}
    print(interim)
    save_path = save_dir / 'results.pth'
    print('save into {:}'.format(save_path))
    torch.save(all_info, save_path)

    wandb_auth()
    wandb.init(project="NAS", group="BOHB")
    wandb.config.update(args)
    wandb.log(interim)
  else:
    main(args, api)
