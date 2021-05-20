#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.07 #
#####################################################
import torch, random
import torch.nn as nn
from copy import deepcopy
from typing import Text
from torch.distributions.categorical import Categorical
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import math, sys
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from ..cell_operations import ResNetBasicblock, drop_path
from .search_cells     import NAS201SearchCell as SearchCell
from .genotypes        import Structure
from nats_bench   import create

CellStructure = Structure
class ArchSampler():
  def __init__(self, api, model, mode="size", prefer="highest", dataset="cifar10", op_names=None, max_nodes=4, search_space="nats-bench"):
    self.db = None
    self.model = model
    self.api = api
    self.mode = mode
    self.prefer = prefer
    self.dataset = dataset
    self.op_names = op_names
    self.max_nodes = max_nodes
    self.search_space = search_space
    self.stats = defaultdict(int)
    self.archs = None # Going to contain List of arch strings

    if mode is None or mode == "perf":
      print("Instantiating ArchSampler with mode=None! This is changed to mode=perf for the purpose of loading recorded architectures")
      mode = "perf"
    try:
      self.load_arch_db(mode if mode in ["perf", "size"] else "perf", prefer)

    except Exception as e:
      print(f"Failed to load arch DB dict with the necessary sampling information due to {e}! Will generate from scratch")
      db = self.generate_arch_dicts(mode=mode)
      self.process_db(db, prefer)

    self.evenly_archs = None
    self.evenly_metrics = None
    self.evenly_sampling_weights = None
    self.evenly_count = None

  def random_topology_func(self, op_names=None, max_nodes=4, ith_candidate=None, fixed_paths=None):
    # Return a random architecture
    if op_names is None:
      op_names = self.op_names
    if max_nodes is None:
      max_nodes = self.max_nodes
    genotypes = []
    for i in range(1, max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if ith_candidate is None and fixed_paths is None:
          op_name  = random.choice( op_names )
        elif ith_candidate is not None:
          assert fixed_paths is not None, "Need to provide pre-sampled paths through the DAG in order to do FairNAS"
          op_name = op_names[fixed_paths[i-1][j][ith_candidate]]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )

  def load_arch_db(self, mode, prefer):
      if self.dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        fname = f'./configs/nas-benchmark/percentiles/{mode}_all_dict_{self.dataset}.pkl'
        with open(fname, 'rb') as f:
          db = pickle.load(f)
      else:
        fname = f'./configs/nas-benchmark/percentiles/{mode}_all_dict.pkl'
        with open(fname, 'rb') as f:
          db = pickle.load(f)
      self.process_db(db, prefer)
      print(f"Loaded {fname} successfully with db len={len(db)}")


  def process_db(self, db, prefer):
      """Calculates weights for non-uniform sampling of architectures"""
      self.db = sorted(list(db.items()), key = lambda x: x[1]) #list of (arch_str, metric) pairs. This will be sorted in ASCENDING order (= the highest perf is at the end!).
      if prefer == "highest":
        sampling_weights = [x[1] for x in self.db]
        total_sampling_weights = sum(sampling_weights)
        sampling_weights = [x/total_sampling_weights for x in sampling_weights] # normalize the probability distribution
      elif prefer == "lowest":
        sampling_weights = [x[1] for x in self.db]
        total_sampling_weights = sum([1/x for x in [item[1] for item in self.db]])
        sampling_weights = [1/x/total_sampling_weights for x in sampling_weights]
      else: # The uniformly random branch
        sampling_weights = [1/len(self.db) for _ in self.db]

      self.sampling_weights = sampling_weights
      self.archs = [x[0] for x in self.db]
      self.metrics = [x[1] for x in self.db]

  def sample(self, mode = "random", perf_percentile = None, size_percentile = None, candidate_num=1, subset=None):
    assert self.sampling_weights[0] == 1/len(self.db) or self.prefer is not None, "If there is no preference, the sampling weights should be uniform"
    assert subset is None or mode != "evenly_split", "Not implemented yet"
    if subset is None:
      all_archs = self.archs
      sampling_weights = self.sampling_weights
    else:
      all_archs = subset
      sampling_weights = self.sampling_weights

    if mode == "evenly_split":
      # NOTE rewrties all_archs again to only be from this evenly_split subset
      assert candidate_num is not None and candidate_num != 1 
      if self.evenly_count is None or self.evenly_count != candidate_num:
        evenly_archs = []
        evenly_metrics = []
        evenly_sampling_weights = []
        chunk_size = math.floor(len(all_archs)/candidate_num)
        for i in range(0, len(all_archs), chunk_size):
          evenly_archs.append(all_archs[min(i+chunk_size, len(all_archs)-1)]) # Like this, we get the best arch from each chunk since it is already sorted by performance if self.mode=perf
          if all_archs == self.archs:
            evenly_metrics.append(self.metrics[min(i+chunk_size, len(all_archs)-1)])
            evenly_sampling_weights.append(self.sampling_weights[min(i+chunk_size, len(all_archs)-1)])
        evenly_archs = [Structure.str2structure(arch) for arch in evenly_archs]
        if all_archs == self.archs:
          print(f"Evenly_split sampled archs (len={len(evenly_archs)}) {[self.api.archstr2index[arch.tostr()] for arch in evenly_archs[0:10]]} from all_archs (len={len(all_archs)}) with chunk_size={chunk_size} and performances head (note this should be average perf across all datasets!) = {evenly_metrics[-5:]}")
        else:
          print(f"Evenly_split sampled archs (len={len(evenly_archs)}) with chunk_size={chunk_size}")

        self.evenly_archs, self.evenly_metrics, self.evenly_sampling_weights, self.evenly_count = evenly_archs, evenly_metrics, evenly_sampling_weights, candidate_num

      all_archs = self.evenly_archs
      sampling_weights = self.evenly_sampling_weights

    if mode == "random":
      if perf_percentile is not None:
        assert self.mode == "perf"
        archs = random.choices(all_archs[round(perf_percentile*len(all_archs)):], weights = sampling_weights, k = candidate_num)
      elif size_percentile is not None:
        assert self.mode == "size"
        archs = random.choices(all_archs[round(size_percentile*len(all_archs)):], weights = sampling_weights, k = candidate_num)
      elif all_archs is None:
        archs = [self.random_topology_func(self.op_names) for _ in range(candidate_num)]
      elif all_archs is not None:
        archs = random.choices(all_archs, weights = sampling_weights, k = candidate_num)
        archs = [Structure.str2structure(arch) for arch in archs]
    elif mode == "quartiles":
      try:
        percentiles = [0, 0.25, 0.50, 0.75, 1]
        assert candidate_num == 4
        archs = [random.choices(all_archs[round(percentiles[i]*len(all_archs)):round(percentiles[i+1]*len(all_archs))])[0] for i in range(len(percentiles)-1)]
        archs = [Structure.str2structure(arch) for arch in archs]
      except Exception as e:
        print(f"Failed to do quartile sampling due to {e}")
        print(percentiles)
        print(all_archs)
    elif mode == "evenly_split":
      assert self.mode == "perf" or self.mode == "size"
      assert all_archs == evenly_archs
      archs = random.choices(evenly_archs, k = candidate_num)
    elif mode == "fairnas":
      assert subset is None, "Not implemented yet" 
      assert candidate_num == len(self.op_names), f"Need to be sampling {len(self.op_names)} paths at a time but instead did {candidate_num}" # Cannot do FairNAS if we do not sample op_names paths at a time - otherwise, it is just normal multi-path sampling
      base_op_order = range(len(self.op_names))
      fixed_paths = [[] for _ in range(1, self.max_nodes)] # Will have i*j paths total in the DAG - see search_cells.py -> forward_urs() for how the sampling is done at forward-pass time
      for i in range(1, self.max_nodes):
        sampled_paths = [random.sample(base_op_order, len(base_op_order)) for _ in range(i)]
        fixed_paths[i-1].extend(sampled_paths)
      archs = [self.random_topology_func(op_names=self.op_names, max_nodes=self.max_nodes, ith_candidate = cand_num, fixed_paths = fixed_paths) for cand_num in range(candidate_num)]
    
    for arch in archs:
      self.stats[self.api.archstr2index[arch.tostr()]] += 1
      
    return archs

  def generate_arch_dicts(self, mode="perf"):
    archs = Structure.gen_all(self.model._op_names, self.model._max_nodes, False)
    api = self.api
    file_suffix = "_percentile.pkl" if mode == "size" else "_perf_percentile.pkl"
    characteristic = "size" if mode == "size" is not None else "perf"
    datasets_archs = {"cifar10" : [], "cifar100": [], "ImageNet16-120" : []}
    new_archs= []

    if mode == "size":
      # Sorted in ascending order
      for i in tqdm(range(len(archs)), desc = f"Loading archs to calculate their {mode} characteristics"):
        new_archs.append((archs[i], api.get_cost_info(api.query_index_by_arch(archs[i]), "cifar10")['params']))
        for dataset in datasets_archs.keys():
          non_avg_results = api.get_cost_info(api.query_index_by_arch(archs[i]), dataset)
          datasets_archs[dataset].append((archs[i], non_avg_results['params']))
        if i % 1000 == 0: # Can take too much memory to keep reusing the same API until we load all 15k archs
          api = create(None, 'topology', fast_mode=True, verbose=False)
      
    elif mode == "perf":
      # Sorted in ascending order
      for i in tqdm(range(len(archs)), desc = f"Loading archs to calculate their {mode} characteristics"):
        new_archs.append((archs[i], summarize_results_by_dataset(genotype=archs[i], api=api, avg_all=True)["avg"]))
        non_avg_results = summarize_results_by_dataset(genotype=archs[i], api=api, avg_all=False)
        for dataset in datasets_archs.keys():
          datasets_archs[dataset].append((archs[i], non_avg_results[dataset]))
        if i % 1000 == 0:
          api = create(None, 'topology', fast_mode=True, verbose=False)
    archs = sorted(new_archs, key=lambda x: x[1])
    for dataset in datasets_archs:
      datasets_archs[dataset] = sorted(datasets_archs[dataset], key = lambda x: x[1])

    datasets_archs = {k: {arch.tostr():metric for arch, metric in v} for k,v in datasets_archs.items()}
    desired_form = {x[0].tostr():x[1] for x in new_archs}

    try:
      with open(f'./configs/nas-benchmark/percentiles/{characteristic}_all_dict.pkl', 'wb') as f:
        pickle.dump(desired_form, f)
      print(f"Saved arch dict to ./configs/nas-benchmark/percentiles/{characteristic}_all_dict.pkl")
      for dataset in datasets_archs.keys():
        with open(f'./configs/nas-benchmark/percentiles/{characteristic}_all_dict_{dataset}.pkl', 'wb') as f:
          pickle.dump(datasets_archs[dataset], f)
        print(f"Saved arch dict to ./configs/nas-benchmark/percentiles/{characteristic}_all_dict_{dataset}.pkl")
    except Exception as e:
      print(f"Failed to save {characteristic} all dict or one of the dataset-specific dicts due to {e}")

    return desired_form

class Controller(nn.Module):
  # we refer to https://github.com/TDeVries/enas_pytorch/blob/master/models/controller.py
  def __init__(self, edge2index, op_names, max_nodes, lstm_size=32, lstm_num_layers=2, tanh_constant=2.5, temperature=5.0):
    super(Controller, self).__init__()
    # assign the attributes
    self.max_nodes = max_nodes
    self.num_edge  = len(edge2index)
    self.edge2index = edge2index
    self.num_ops   = len(op_names)
    self.op_names  = op_names
    self.lstm_size = lstm_size
    self.lstm_N    = lstm_num_layers
    self.tanh_constant = tanh_constant
    self.temperature   = temperature
    # create parameters
    self.register_parameter('input_vars', nn.Parameter(torch.Tensor(1, 1, lstm_size)))
    self.w_lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size, num_layers=self.lstm_N)
    self.w_embd = nn.Embedding(self.num_ops, self.lstm_size)
    self.w_pred = nn.Linear(self.lstm_size, self.num_ops)

    nn.init.uniform_(self.input_vars         , -0.1, 0.1)
    nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
    nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)
    nn.init.uniform_(self.w_embd.weight      , -0.1, 0.1)
    nn.init.uniform_(self.w_pred.weight      , -0.1, 0.1)

  def convert_structure(self, _arch):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_index = _arch[self.edge2index[node_str]]
        op_name  = self.op_names[op_index]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure(genotypes)

  def forward(self):

    inputs, h0 = self.input_vars, None
    log_probs, entropys, sampled_arch = [], [], []
    for iedge in range(self.num_edge):
      outputs, h0 = self.w_lstm(inputs, h0)
      
      logits = self.w_pred(outputs)
      logits = logits / self.temperature
      logits = self.tanh_constant * torch.tanh(logits)
      # distribution
      op_distribution = Categorical(logits=logits)
      op_index    = op_distribution.sample()
      sampled_arch.append( op_index.item() )

      op_log_prob = op_distribution.log_prob(op_index)
      log_probs.append( op_log_prob.view(-1) )
      op_entropy  = op_distribution.entropy()
      entropys.append( op_entropy.view(-1) )
      
      # obtain the input embedding for the next step
      inputs = self.w_embd(op_index)
    return torch.sum(torch.cat(log_probs)), torch.sum(torch.cat(entropys)), self.convert_structure(sampled_arch)

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

class GenericNAS201Model(nn.Module):

  def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats, arch_sampler = None):
    super(GenericNAS201Model, self).__init__()
    self._C          = C
    self._layerN     = N
    self._max_nodes  = max_nodes
    self._stem       = nn.Sequential(
                         nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                         nn.BatchNorm2d(C))
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
    print(f"Initializing model with total number of cells={len(layer_channels)}")
    C_prev, num_edge, edge2index = C, None, None
    self._cells      = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self._cells.append(cell)
      C_prev = cell.out_dim
    self._op_names   = deepcopy(search_space)
    self._Layer      = len(self._cells)
    self.edge2index  = edge2index
    self.lastact     = nn.Sequential(nn.BatchNorm2d(C_prev, affine=affine, track_running_stats=track_running_stats), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier  = nn.Linear(C_prev, num_classes)
    self._num_edge   = num_edge
    # algorithm related
    self.arch_parameters = nn.Parameter(1e-3*torch.randn(num_edge, len(search_space)))
    self._mode        = None
    self.dynamic_cell = None
    self._tau         = None
    self._algo        = None
    self._drop_path   = None
    self.verbose      = False
    self.logits_only = False
    self.arch_sampler = arch_sampler

  def arch_params(self):
      for n,p in self.named_parameters():
          if 'arch' in n and p.requires_grad:
              yield p
          else:
              continue
  def weight_params(self):
    for n,p in self.named_parameters():
        if 'arch' not in n and p.requires_grad:
            yield p
        else:
            continue
  def set_algo(self, algo: Text):
    # used for searching
    assert self._algo is None, 'This functioin can only be called once.'
    self._algo = algo
    if algo == 'enas':
      self.controller = Controller(self.edge2index, self._op_names, self._max_nodes)
    else:
      self.arch_parameters = nn.Parameter( 1e-3*torch.randn(self._num_edge, len(self._op_names)) )
      if algo == 'gdas':
        self._tau         = 10
    
  def set_cal_mode(self, mode, dynamic_cell=None, sandwich_cells=None):
    assert mode in ['gdas', 'enas', 'urs', 'joint', 'select', 'dynamic', 'sandwich']
    self._mode = mode
    if mode == 'dynamic': self.dynamic_cell = deepcopy(dynamic_cell)
    else                : self.dynamic_cell = None
    if mode == "sandwich":
      assert sandwich_cells is not None
      self.sandwich_cells = sandwich_cells

  def set_drop_path(self, progress, drop_path_rate):
    if drop_path_rate is None:
      self._drop_path = None
    elif progress is None:
      self._drop_path = drop_path_rate
    else:
      self._drop_path = progress * drop_path_rate

  @property
  def mode(self):
    return self._mode

  @property
  def drop_path(self):
    return self._drop_path

  @property
  def weights(self):
    xlist = list(self._stem.parameters())
    xlist+= list(self._cells.parameters())
    xlist+= list(self.lastact.parameters())
    xlist+= list(self.global_pooling.parameters())
    xlist+= list(self.classifier.parameters())
    return xlist

  def set_tau(self, tau):
    self._tau = tau

  @property
  def tau(self):
    return self._tau

  @property
  def alphas(self):
    if self._algo == 'enas':
      return list(self.controller.parameters())
    else:
      return [self.arch_parameters]

  @property
  def message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self._cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self._cells), cell.extra_repr())
    return string

  def show_alphas(self):
    with torch.no_grad():
      if self._algo == 'enas':
        return 'w_pred :\n{:}'.format(self.controller.w_pred.weight)
      else:
        return 'arch-parameters :\n{:}'.format(nn.functional.softmax(self.arch_parameters, dim=-1).cpu())
          

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={_max_nodes}, N={_layerN}, L={_Layer}, alg={_algo})'.format(name=self.__class__.__name__, **self.__dict__))

  @property
  def genotype(self):
    genotypes = []
    for i in range(1, self._max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = self.arch_parameters[ self.edge2index[node_str] ]
          op_name = self._op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append(tuple(xlist))
    return Structure(genotypes)

  def dync_genotype(self, use_random=False):
    genotypes = []
    with torch.no_grad():
      alphas_cpu = nn.functional.softmax(self.arch_parameters, dim=-1)
    for i in range(1, self._max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if use_random:
          op_name  = random.choice(self._op_names)
        else:
          weights  = alphas_cpu[ self.edge2index[node_str] ]
          op_index = torch.multinomial(weights, 1).item()
          op_name  = self._op_names[ op_index ]
        xlist.append((op_name, j))
      genotypes.append(tuple(xlist))
    return Structure(genotypes)

  def get_log_prob(self, arch):
    with torch.no_grad():
      logits = nn.functional.log_softmax(self.arch_parameters, dim=-1)
    select_logits = []
    for i, node_info in enumerate(arch.nodes):
      for op, xin in node_info:
        node_str = '{:}<-{:}'.format(i+1, xin)
        op_index = self._op_names.index(op)
        select_logits.append( logits[self.edge2index[node_str], op_index] )
    return sum(select_logits).item()

  def generate_arch_all_dicts(self, api, size_percentile = None, perf_percentile = None):
    archs = Structure.gen_all(self._op_names, self._max_nodes, False)
    pairs = [(self.get_log_prob(arch), arch) for arch in archs]
    # TODO should get rid of the size_percentile/perf_percentile_all_dict.pkl files - can just use the all_arch_dict
    if size_percentile is not None or perf_percentile is not None:

      file_suffix = "_percentile.pkl" if size_percentile is not None else "_perf_percentile.pkl"
      characteristic = "size" if size_percentile is not None else "perf"

      try:
        from pathlib import Path
        with open(f'./configs/nas-benchmark/percentiles/{perf_percentile}{file_suffix}', 'rb') as f:
          archs=pickle.load(f)
        print(f"Succeeded in loading architectures from ./configs/nas-benchmark/percentiles/{perf_percentile}{file_suffix}! We have archs with len={len(archs)}.")
        if len(archs) == 0:
          print(f"Len of loaded archs is 0! Must restart, RIP")
          raise NotImplementedError
      except Exception as e:
        print(f"Couldnt load the percentiles! Will calculate them from scratch. Exception {e}")
        if size_percentile is not None:
          # Sorted in ascending order
          new_archs= []
          for i in range(len(archs)):
            new_archs.append((archs[i], api.get_cost_info(api.query_index_by_arch(archs[i]), dataset if dataset != "cifar5m" else "cifar10")['params']))
            if i % 1500 == 0: # Can take too much memory to keep reusing the same API until we load all 15k archs
              api = create(None, 'topology', fast_mode=True, verbose=False)
          
          archs = sorted(new_archs, key=lambda x: x[1])
          archs = archs[round(size_percentile*len(archs)):]
          try:
            from pathlib import Path
            Path('./configs/nas-benchmark/percentiles/').mkdir(parents=True, exist_ok=True)
            with open(f'./configs/nas-benchmark/percentiles/{size_percentile}{file_suffix}', 'wb') as f:
              pickle.dump(archs, f)
          except Exception as e:
            print(f"Couldnt save the percentiles! Exception {e}")
        elif perf_percentile is not None:
          # Sorted in ascending order
          new_archs= []
          for i in range(len(archs)):
            new_archs.append((archs[i], summarize_results_by_dataset(genotype=archs[i], api=api, avg_all=True)["avg"]))
            if i % 1500 == 0:
              api = create(None, 'topology', fast_mode=True, verbose=False)
          archs = sorted(new_archs, key=lambda x: x[1])
          archs = archs[round(perf_percentile*len(archs)):]
          try:
            from pathlib import Path
            Path('./configs/nas-benchmark/percentiles/').mkdir(parents=True, exist_ok=True)
            with open(f'./configs/nas-benchmark/percentiles/{size_percentile}{file_suffix}', 'wb') as f:
              pickle.dump(archs, f)
          except Exception as e:
            print(f"Couldnt save the percentiles! Exception {e}")  

      try:
        with open(f'./configs/nas-benchmark/percentiles/{characteristic}_all_dict.pkl', 'wb') as f:
          pickle.dump({x[0].tostr():x[1] for x in new_archs}, f)
      except:
        print(f"Failed to save {characteristic} all dict")

      characteristics_only = [a[1] for a in archs]
      avg_characteristic = sum(characteristics_only)/len(archs)
      archs_min, archs_max = min(characteristics_only), max(characteristics_only)
      print(f"Limited all archs to {len(archs)} architectures with average {characteristic} {avg_characteristic}, min={archs_min}, max={archs_max}")
      archs = [a[0] for a in archs]

    return archs
      
  def return_topK(self, K, use_random=False, size_percentile=None, perf_percentile=None, api=None, dataset=None):
    """NOTE additionaly outputs perf/size_all_dict.pkl mainly with shape {arch_str: perf_metric} """
    if size_percentile is not None or perf_percentile is not None:
      characteristic = "size" if size_percentile is not None else "perf"

      archs = self.generate_all_arch_dicts(size_percentile = size_percentile, perf_percentile = perf_percentile, api = api)

      characteristics_only = [a[1] for a in archs]
      avg_characteristic = sum(characteristics_only)/len(archs)
      archs_min, archs_max = min(characteristics_only), max(characteristics_only)
      print(f"Limited all archs to {len(archs)} architectures with average {characteristic} {avg_characteristic}, min={archs_min}, max={archs_max}")
      archs = [a[0] for a in archs]

    if use_random:
      if self.xargs.search_space_paper == "nats-bench":
        archs = Structure.gen_all(self._op_names, self._max_nodes, False)
        pairs = [(self.get_log_prob(arch), arch) for arch in archs]
        if K < 0 or K >= len(archs): K = len(archs)
        sampled = random.sample(archs, K)
      elif self.xargs.search_space_paper == "darts":
        sampled = self.arch_sampler.sample(mode="random", candidate_num=K)

      return sampled
    else:
      if self.xargs.search_space_paper in ["nats-bench"]:
        archs = Structure.gen_all(self._op_names, self._max_nodes, False)
        pairs = [(self.get_log_prob(arch), arch) for arch in archs]
        if K < 0 or K >= len(archs): K = len(archs)
        sorted_pairs = sorted(pairs, key=lambda x: -x[0])
        return_pairs = [sorted_pairs[_][1] for _ in range(K)]
      else:
        return_pairs = self.arch_sampler.sample(mode="random", candidate_num=K)
      return return_pairs

  def normalize_archp(self):
    if self.mode == 'gdas':
      while True:
        gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
        logits  = (self.arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
        probs   = nn.functional.softmax(logits, dim=1)
        index   = probs.max(-1, keepdim=True)[1]
        one_h   = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        hardwts = one_h - probs.detach() + probs
        if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
          continue
        else: break
      with torch.no_grad():
        hardwts_cpu = hardwts.detach().cpu()
      return hardwts, hardwts_cpu, index, 'GUMBEL'
    else:
      alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)
      index   = alphas.max(-1, keepdim=True)[1]
      with torch.no_grad():
        alphas_cpu = alphas.detach().cpu()
      return alphas, alphas_cpu, index, 'SOFTMAX'

  def forward(self, inputs):
    alphas, alphas_cpu, index, verbose_str = self.normalize_archp()
    feature = self._stem(inputs)
    for i, cell in enumerate(self._cells):
      if isinstance(cell, SearchCell):
        if self.mode == 'urs':
          feature = cell.forward_urs(feature)
          if self.verbose:
            verbose_str += '-forward_urs'
        elif self.mode == 'select':
          feature = cell.forward_select(feature, alphas_cpu)
          if self.verbose:
            verbose_str += '-forward_select'
        elif self.mode == 'joint':
          feature = cell.forward_joint(feature, alphas)
          if self.verbose:
            verbose_str += '-forward_joint'
        elif self.mode == 'dynamic':
          feature = cell.forward_dynamic(feature, self.dynamic_cell)
          if self.verbose:
            verbose_str += '-forward_dynamic'
        elif self.mode == "sandwich":
          sandwich_cell = random.sample(self.sandwich_cells, 1)[0]
          feature = cell.forward_dynamic(feature, sandwich_cell)
        elif self.mode == 'gdas':
          feature = cell.forward_gdas(feature, alphas, index)
          if self.verbose:
            verbose_str += '-forward_gdas'
        else: raise ValueError('invalid mode={:}'.format(self.mode))
      else: feature = cell(feature)
      if self.drop_path is not None:
        feature = drop_path(feature, self.drop_path)
    if self.verbose and random.random() < 0.001:
      print(verbose_str)
    out = self.lastact(feature)
    out = self.global_pooling(out)
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)
    if not self.logits_only:
      return out, logits
    else:
      return logits
