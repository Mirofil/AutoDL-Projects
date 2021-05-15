####################
# DARTS, ICLR 2019 #
####################
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Text, Dict
from .search_cells import NASNetSearchCell as SearchCell
from .genotypes        import Structure
import random

# The macro structure is based on NASNet
class NASNetworkDARTS(nn.Module):

  def __init__(self, C: int, N: int, steps: int, multiplier: int, stem_multiplier: int,
               num_classes: int, search_space: List[Text], affine: bool, track_running_stats: bool):
    super(NASNetworkDARTS, self).__init__()
    self._C        = C
    self._layerN   = N
    self._steps    = steps
    self._multiplier = multiplier
    self._stem = nn.Sequential(
                    nn.Conv2d(3, C*stem_multiplier, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C*stem_multiplier))
  
    # config for each layer
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    num_edge, edge2index = None, None
    C_prev_prev, C_prev, C_curr, reduction_prev = C*stem_multiplier, C*stem_multiplier, C, False

    self._cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
      if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
      else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self._cells.append( cell )
      C_prev_prev, C_prev, reduction_prev = C_prev, multiplier*C_curr, reduction
    self._op_names   = deepcopy( search_space )
    self._Layer     = len(self._cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.arch_normal_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )
    self.arch_reduce_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )

    # NOT USED HERE - taken from NAS201 model for compatibility
    self._mode        = None
    self.dynamic_cell = None
    self._tau         = None
    self._algo        = None
    self._drop_path   = None
    self.verbose      = False
    self.logits_only = False
    self.arch_sampler = None
    self._max_nodes = steps # TODO should be the same I think?


  def get_weights(self) -> List[torch.nn.Parameter]:
    xlist = list( self._stem.parameters() ) + list( self._cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def get_alphas(self) -> List[torch.nn.Parameter]:
    return [self.arch_normal_parameters, self.arch_reduce_parameters]

  def show_alphas(self) -> Text:
    with torch.no_grad():
      A = 'arch-normal-parameters :\n{:}'.format( nn.functional.softmax(self.arch_normal_parameters, dim=-1).cpu() )
      B = 'arch-reduce-parameters :\n{:}'.format( nn.functional.softmax(self.arch_reduce_parameters, dim=-1).cpu() )
    return '{:}\n{:}'.format(A, B)

  def get_message(self) -> Text:
    string = self.extra_repr()
    for i, cell in enumerate(self._cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self._cells), cell.extra_repr())
    return string

  def extra_repr(self) -> Text:
    return ('{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  @property
  def genotype(self) -> Dict[Text, List]:
    def _parse(weights):
      gene = []
      for i in range(self._steps):
        edges = []
        for j in range(2+i):
          node_str = '{:}<-{:}'.format(i, j)
          ws = weights[ self.edge2index[node_str] ]
          for k, op_name in enumerate(self._op_names):
            if op_name == 'none': continue
            edges.append( (op_name, j, ws[k]) )
        # (TODO) xuanyidong:
        # Here the selected two edges might come from the same input node.
        # And this case could be a problem that two edges will collapse into a single one
        # due to our assumption -- at most one edge from an input node during evaluation.
        edges = sorted(edges, key=lambda x: -x[-1])
        selected_edges = edges[:2]
        gene.append( tuple(selected_edges) )
      return gene
    with torch.no_grad():
      gene_normal = _parse(torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy())
      gene_reduce = _parse(torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy())
    return {'normal': gene_normal, 'normal_concat': list(range(2+self._steps-self._multiplier, self._steps+2)),
            'reduce': gene_reduce, 'reduce_concat': list(range(2+self._steps-self._multiplier, self._steps+2))}

  def forward(self, inputs):

    normal_w = nn.functional.softmax(self.arch_normal_parameters, dim=1)
    reduce_w = nn.functional.softmax(self.arch_reduce_parameters, dim=1)

    s0 = s1 = self._stem(inputs)
    for i, cell in enumerate(self._cells):
      if cell.reduction: ww = reduce_w
      else             : ww = normal_w
      s0, s1 = s1, cell.forward_darts(s0, s1, ww)
    out = self.lastact(s1)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits

  def set_algo(self, algo: Text):
    pass
    
  def set_cal_mode(self, mode, dynamic_cell=None, sandwich_cells=None):
    pass

  def set_drop_path(self, progress, drop_path_rate):
    pass

  @property
  def mode(self):
    return self._mode

  @property
  def drop_path(self):
    return self._drop_path

  @property
  def weights(self):
    return self.get_weights()

  @property
  def alphas(self):
    return self.get_alphas()

  @property
  def message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self._cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self._cells), cell.extra_repr())
    return string

  def show_alphas(self):
    with torch.no_grad():
      return 'arch-parameters :\n{:}'.format(nn.functional.softmax(self.arch_parameters, dim=-1).cpu())

  # @property
  # def genotype(self):
  #   genotypes = []
  #   for i in range(1, self._max_nodes):
  #     xlist = []
  #     for j in range(i):
  #       node_str = '{:}<-{:}'.format(i, j)
  #       with torch.no_grad():
  #         weights = self.arch_parameters[ self.edge2index[node_str] ]
  #         op_name = self._op_names[ weights.argmax().item() ]
  #       xlist.append((op_name, j))
  #     genotypes.append(tuple(xlist))
  #   return Structure(genotypes)

  # TODO used for other algos only?
  # def dync_genotype(self, use_random=False):
  #   genotypes = []
  #   with torch.no_grad():
  #     alphas_cpu = nn.functional.softmax(self.arch_parameters, dim=-1)
  #   for i in range(1, self._max_nodes):
  #     xlist = []
  #     for j in range(i):
  #       node_str = '{:}<-{:}'.format(i, j)
  #       if use_random:
  #         op_name  = random.choice(self._op_names)
  #       else:
  #         weights  = alphas_cpu[ self.edge2index[node_str] ]
  #         op_index = torch.multinomial(weights, 1).item()
  #         op_name  = self._op_names[ op_index ]
  #       xlist.append((op_name, j))
  #     genotypes.append(tuple(xlist))
  #   return Structure(genotypes)

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
    # if self.mode == 'gdas':
    #   while True:
    #     gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
    #     logits  = (self.arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
    #     probs   = nn.functional.softmax(logits, dim=1)
    #     index   = probs.max(-1, keepdim=True)[1]
    #     one_h   = torch.zeros_like(logits).scatter_(-1, index, 1.0)
    #     hardwts = one_h - probs.detach() + probs
    #     if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
    #       continue
    #     else: break
    #   with torch.no_grad():
    #     hardwts_cpu = hardwts.detach().cpu()
    #   return hardwts, hardwts_cpu, index, 'GUMBEL'
    # else:
    alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)
    index   = alphas.max(-1, keepdim=True)[1]
    with torch.no_grad():
      alphas_cpu = alphas.detach().cpu()
    return alphas, alphas_cpu, index, 'SOFTMAX'