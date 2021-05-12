##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
##################################################################
# Regularized Evolution for Image Classifier Architecture Search #
##################################################################
# python ./exps/NATS-algos/regularized_ea.py --dataset cifar10 --search_space tss --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --loops_if_rand 100 --hp 200 --epoch 50 
# python ./exps/NATS-algos/regularized_ea.py --dataset cifar100 --search_space tss --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/NATS-algos/regularized_ea.py --dataset ImageNet16-120 --search_space tss --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/NATS-algos/regularized_ea.py --dataset cifar10 --search_space sss --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/NATS-algos/regularized_ea.py --dataset cifar100 --search_space sss --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/NATS-algos/regularized_ea.py --dataset ImageNet16-120 --search_space sss --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --rand_seed 1
# python ./exps/NATS-algos/regularized_ea.py  --dataset ${dataset} --search_space ${search_space} --time_budget ${time_budget} --ea_cycles 200 --ea_population 10 --ea_sample_size 3 --use_proxy 0
##################################################################
from lib.utils.train_loop import get_finetune_scheduler
from lib.utils.sotl_utils import eval_archs_on_batch
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
from models       import CellStructure, get_search_spaces
from nats_bench   import create
from utils.sotl_utils import simulate_train_eval_sotl, query_all_results_by_arch, wandb_auth
import wandb


class Model(object):

  def __init__(self):
    self.arch = None
    self.accuracy = None
    
  def __str__(self):
    """Prints a readable version of this bitstring."""
    return '{:}'.format(self.arch)
  

def random_topology_func(op_names, max_nodes=4):
  # Return a random architecture
  def random_architecture():
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name  = random.choice( op_names )
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
  return random_architecture


def random_size_func(info):
  # Return a random architecture
  def random_architecture():
    channels = []
    for i in range(info['numbers']):
      channels.append(
        str(random.choice(info['candidates'])))
    return ':'.join(channels)
  return random_architecture


def mutate_topology_func(op_names):
  """Computes the architecture for a child of the given parent architecture.
  The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
  """
  def mutate_topology_func(parent_arch):
    child_arch = deepcopy( parent_arch )
    node_id = random.randint(0, len(child_arch.nodes)-1)
    node_info = list( child_arch.nodes[node_id] )
    snode_id = random.randint(0, len(node_info)-1)
    xop = random.choice( op_names )
    while xop == node_info[snode_id][0]:
      xop = random.choice( op_names )
    node_info[snode_id] = (xop, node_info[snode_id][1])
    child_arch.nodes[node_id] = tuple( node_info )
    return child_arch
  return mutate_topology_func


def mutate_size_func(info):
  """Computes the architecture for a child of the given parent architecture.
  The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
  """
  def mutate_size_func(parent_arch):
    child_arch = deepcopy(parent_arch)
    child_arch = child_arch.split(':')
    index = random.randint(0, len(child_arch)-1)
    child_arch[index] = str(random.choice(info['candidates']))
    return ':'.join(child_arch)
  return mutate_size_func


def regularized_evolution(cycles, population_size, sample_size, time_budget, random_arch, mutate_arch, api, use_proxy, dataset, xargs):
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
  population = collections.deque()
  api.reset_time()
  history, total_time_cost = [], []  # Not used by the algorithm, only used to report results.
  current_best_index = []
  # if use_proxy:
  #   xargs.hp = '12'
  # Initialize the population with random models.
  while len(population) < population_size:
    model = Model()
    model.arch = random_arch()

    model.accuracy, _, _, total_cost = simulate_train_eval_sotl(api,
      model.arch, dataset, hp=xargs.hp, iepoch=xargs.epoch, metric=xargs.metric, e=xargs.e)
    # Append the info
    population.append(model)
    history.append((model.accuracy, model.arch))
    total_time_cost.append(total_cost)
    current_best_index.append(api.query_index_by_arch(max(history, key=lambda x: x[0])[1]))

  # Carry out evolution in cycles. Each cycle produces a model and removes another.
  while total_time_cost[-1] < time_budget:
    # Sample randomly chosen models from the current population.
    start_time, sample = time.time(), []
    while len(sample) < sample_size:
      # Inefficient, but written this way for clarity. In the case of neural
      # nets, the efficiency of this line is irrelevant because training neural
      # nets is the rate-determining step.
      candidate = random.choice(list(population))
      sample.append(candidate)

    # The parent is the best model in the sample.
    parent = max(sample, key=lambda i: i.accuracy)

    # Create the child model and store it.
    child = Model()
    child.arch = mutate_arch(parent.arch)
    child.accuracy, _, _, total_cost = simulate_train_eval_sotl(api,
      child.arch, dataset, hp=xargs.hp, iepoch=xargs.epoch, metric=xargs.metric, e=xargs.e)
    # Append the info
    population.append(child)
    history.append((child.accuracy, child.arch))
    current_best_index.append(api.query_index_by_arch(max(history, key=lambda x: x[0])[1]))
    total_time_cost.append(total_cost)

    # Remove the oldest model.
    population.popleft()
  return history, current_best_index, total_time_cost


def regularized_evolution_ws(network, train_loader, population_size, sample_size, mutate_arch, cycles, arch_sampler, api, dataset, xargs, train_steps=15, metric="loss"):
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
  history, total_time_cost = [], []  # Not used by the algorithm, only used to report results.
  cur_best_arch = []

  # Initialize the population with random models.
  while len(population) < population_size:
    model = deepcopy(network)
    w_optimizer, w_scheduler, criterion = get_finetune_scheduler(xargs.scheduler, config, xargs, model, None)

    cur_arch = arch_sampler.random_topology_func()

    model.set_cal_mode("dynamic", cur_arch)

    metrics, sum_metrics = eval_archs_on_batch(xloader=train_loader, archs=[cur_arch], network = network, criterion=criterion, train_steps=train_steps, same_batch=True, metric=metric, train_loader=train_loader, w_optimizer=w_optimizer)
    model.metric = metrics[0]
    model.arch = cur_arch

    # Append the info
    population.append(model)
    history.append((metric, cur_arch))
    history.append({metric: metrics[0], "sum": sum_metrics, "arch": cur_arch})
    # total_time_cost.append(total_cost)
    if xargs.rea_metric in ['loss', 'acc']:
      decision_metric, decision_lambda = metrics[0], lambda x: x[metric][0]
    elif xargs.rea_metric in ['sotl']:
      decision_metric, decision_lambda = sum_metrics["loss"], lambda x: x["sum"]["loss"]
    elif xargs.rea_metric in ['soacc']:
      decision_metric, decision_lambda = sum_metrics["acc"], lambda x: x["sum"]["acc"]
    cur_best_arch.append(max(history, key=decision_lambda)["arch"].tostr())

  # Carry out evolution in cycles. Each cycle produces a model and removes another.
  for i in range(cycles):
    # Sample randomly chosen models from the current population.
    start_time, sample = time.time(), []
    while len(sample) < sample_size:
      # Inefficient, but written this way for clarity. In the case of neural
      # nets, the efficiency of this line is irrelevant because training neural
      # nets is the rate-determining step.
      candidate = random.choice(list(population))
      sample.append(candidate)

    # The parent is the best model in the sample.
    parent = max(sample, key=lambda i: i.metric)

    # Create the child model and store it.
    child = deepcopy(network)
    child.arch = mutate_arch(parent.arch)

    #TODO eval
    # Append the info
    population.append(child)
    history.append((child.metric, child.arch))
    cur_best_arch.append(api.archstr2index[(max(history, key=lambda x: x[0])[1]).tostr()])
    # total_time_cost.append(total_cost)

    # Remove the oldest model.
    population.popleft()

  return history, cur_best_arch, total_time_cost

def main(xargs, api):
  torch.set_num_threads(4)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  search_space = get_search_spaces(xargs.search_space, 'nats-bench')
  if xargs.search_space == 'tss':
    random_arch = random_topology_func(search_space)
    mutate_arch = mutate_topology_func(search_space)
  else:
    random_arch = random_size_func(search_space)
    mutate_arch = mutate_size_func(search_space)

  x_start_time = time.time()
  logger.log('{:} use api : {:}'.format(time_string(), api))
  logger.log('-'*30 + ' start searching with the time budget of {:} s'.format(xargs.time_budget))
  history, current_best_index, total_times = regularized_evolution(xargs.ea_cycles,
                                                                   xargs.ea_population,
                                                                   xargs.ea_sample_size,
                                                                   xargs.time_budget,
                                                                   random_arch, mutate_arch, api, xargs.use_proxy > 0, xargs.dataset, xargs=xargs)
  logger.log('{:} regularized_evolution finish with history of {:} arch with {:.1f} s (real-cost={:.2f} s).'.format(time_string(), len(history), total_times[-1], time.time()-x_start_time))

  best_arch = max(history, key=lambda x: x[0])[1]
  logger.log('{:} best arch is {:}'.format(time_string(), best_arch))
  
  info = api.query_info_str_by_arch(best_arch, '200' if xargs.search_space == 'tss' else '90')
  abridged_results = query_all_results_by_arch(best_arch, api, iepoch=199, hp='200')
  logger.log('{:}'.format(info))
  logger.log('-'*100)
  logger.close()
  return logger.log_dir, current_best_index, total_times, abridged_results


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  parser.add_argument('--search_space',       type=str,   choices=['tss', 'sss'], help='Choose the search space.')
  # hyperparameters for REA
  parser.add_argument('--ea_cycles',          type=int,   help='The number of cycles in EA.')
  parser.add_argument('--ea_population',      type=int,   help='The population size in EA.')
  parser.add_argument('--ea_sample_size',     type=int,   help='The sample size in EA.')
  parser.add_argument('--time_budget',        type=int,   default=20000, help='The total time cost budge for searching (in seconds).')
  parser.add_argument('--use_proxy',          type=int,   default=1,     help='Whether to use the proxy (H0) task or not.')
  #
  parser.add_argument('--loops_if_rand',      type=int,   default=500,   help='The total runs for evaluation.')
  # log
  parser.add_argument('--save_dir',           type=str,   default='./output/search', help='Folder to save checkpoints and log.')
  parser.add_argument('--rand_seed',          type=int,   default=-1,    help='manual seed')
  parser.add_argument('--metric', type=str, default='valid-accuracy', choices=['valid-accuracy', 'train-loss', 'valid-loss'], help='valid-accuracy/train-loss/valid-loss')
  parser.add_argument('--epoch', type=int, default=11, help='12 or 200')
  parser.add_argument('--hp', type=str, default='12', help='12 or 200')
  parser.add_argument('--e', type=int, default=1, help='SOTL-E')
  
  args = parser.parse_args()

  if 'TORCH_HOME' not in os.environ:
    if os.path.exists('/notebooks/storage/.torch/'):
      os.environ["TORCH_HOME"] = '/notebooks/storage/.torch/'
      
  api = create(None, args.search_space, fast_mode=True, verbose=False)

  args.save_dir = os.path.join('{:}-{:}'.format(args.save_dir, args.search_space),
                               '{:}-T{:}{:}'.format(args.dataset, args.time_budget, '' if args.use_proxy > 0 else '-FULL'),
                               'R-EA-SS{:}'.format(args.ea_sample_size))
  print('save-dir : {:}'.format(args.save_dir))
  print('xargs : {:}'.format(args))
  wandb_auth()
  wandb.init(project="NAS", group="REA")
  wandb.config.update(args)
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

    wandb.log(interim)


  else:
    main(args, api)
