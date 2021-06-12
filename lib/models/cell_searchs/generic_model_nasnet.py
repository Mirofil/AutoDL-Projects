####################
# DARTS, ICLR 2019 #
####################
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Text, Dict
from .search_cells import NASNetSearchCell as SearchCell
from .genotypes        import Structure
from collections import namedtuple
import random
import pickle
from nats_bench   import create
import numpy as np
import sys
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from models.cell_operations import DARTS_OPS_STR2IDX



Genotype_tuple = namedtuple('Genotype_tuple', 'normal normal_concat reduce reduce_concat')

class Genotype:
  def __init__(self, normal, normal_concat, reduce, reduce_concat) -> None:
      self.normal = normal
      self.normal_concat = normal_concat
      self.reduce = reduce
      self.reduce_concat = reduce_concat
      self.genotype_tuple = Genotype_tuple(normal, normal_concat, reduce, reduce_concat)
      
  def tostr(self):
    return str(self.genotype_tuple)

  def __hash__(self):
    return hash(str(self.genotype_tuple))
    
  def __repr__(self):
    return str(self.genotype_tuple)
  
  def __getitem__(self, k):
    return getattr(self, k)
  
  
# The macro structure is based on NASNet
class NASNetworkGeneric(nn.Module):

  def __init__(self, C: int, N: int, steps: int, multiplier: int, stem_multiplier: int,
               num_classes: int, search_space: List[Text], affine: bool, track_running_stats: bool):
    super(NASNetworkGeneric, self).__init__()
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

  def random_topology_func(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    # num_ops = len(genotypes.PRIMITIVES)
    num_ops = len(self._op_names)
    # n_nodes = self.model._steps
    n_nodes = self._max_nodes

    normal = []
    reduction = []
    for i in range(n_nodes):
        ops = np.random.choice(range(num_ops), 4)
        nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
        nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
        normal.extend([(self._op_names[ops[0]], nodes_in_normal[0]), (self._op_names[ops[1]], nodes_in_normal[1])])
        reduction.extend([(self._op_names[ops[2]], nodes_in_reduce[0]), (self._op_names[ops[3]], nodes_in_reduce[1])])
    # return (normal, reduction)
    return Genotype(normal = normal, normal_concat = list(range(2+self._steps-self._multiplier, self._steps+2)), 
                reduce = reduction, reduce_concat = list(range(2+self._steps-self._multiplier, self._steps+2)))
  
  def sample_arch(self):
    return self.random_topology_func()

  def get_genotype(self, original_darts_format=True) -> Dict[Text, List]:
    # Used for NASBench 301 purposes among other things
    with torch.no_grad():
      gene_normal = self._parse(torch.softmax(torch.randn_like(self.arch_normal_parameters), dim=-1).cpu().numpy(), original_darts_format=original_darts_format)
      gene_reduce = self._parse(torch.softmax(torch.randn_like(self.arch_reduce_parameters), dim=-1).cpu().numpy(), original_darts_format=original_darts_format)
    return Genotype(normal = gene_normal, normal_concat = list(range(2+self._steps-self._multiplier, self._steps+2)), 
                reduce = gene_reduce, reduce_concat = list(range(2+self._steps-self._multiplier, self._steps+2)))
    # return {'normal': gene_normal, 'normal_concat': list(range(2+self._steps-self._multiplier, self._steps+2)),
    #         'reduce': gene_reduce, 'reduce_concat': list(range(2+self._steps-self._multiplier, self._steps+2))}

  @property
  def arch_parameters(self):
    return self.get_alphas()

  @property
  def genotype(self) -> Dict[Text, List]:

    with torch.no_grad():
      gene_normal = self._parse(torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy())
      gene_reduce = self._parse(torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy())
      
    return Genotype(normal = gene_normal, normal_concat = list(range(2+self._steps-self._multiplier, self._steps+2)), 
                    reduce = gene_reduce, reduce_concat = list(range(2+self._steps-self._multiplier, self._steps+2)))
    # return {'normal': gene_normal, 'normal_concat': list(range(2+self._steps-self._multiplier, self._steps+2)),
    #         'reduce': gene_reduce, 'reduce_concat': list(range(2+self._steps-self._multiplier, self._steps+2))}

  def _parse(self, weights, original_darts_format=False):
    gene = []
    for i in range(self._steps):
      edges = []
      for j in range(2+i):
        node_str = '{:}<-{:}'.format(i, j)
        ws = weights[ self.edge2index[node_str] ]
        for k, op_name in enumerate(self._op_names):
          if op_name == 'none': continue
          if not original_darts_format:
            edges.append( (op_name, j, ws[k]) )
          else:
            edges.append((op_name, j, ws[k]))
      # (TODO) xuanyidong:
      # Here the selected two edges might come from the same input node.
      # And this case could be a problem that two edges will collapse into a single one
      # due to our assumption -- at most one edge from an input node during evaluation.
      edges = sorted(edges, key=lambda x: -x[-1]) # NOTE This is the argmax over softmax coefficients part
      selected_edges = edges[:2]
      if not original_darts_format:
        gene.append( tuple(selected_edges) )
      else:
        gene.extend([(x[0], x[1]) for x in selected_edges])
    return gene

  def set_cal_mode(self, mode, dynamic_cell=None, sandwich_cells=None):
    assert mode in ['gdas', 'enas', 'urs', 'joint', 'select', 'dynamic', 'sandwich']
    self._mode = mode
    if mode == 'dynamic': self.dynamic_cell = deepcopy(dynamic_cell)
    else                : self.dynamic_cell = None
    if mode == "sandwich":
      assert sandwich_cells is not None
      self.sandwich_cells = sandwich_cells
      
  @staticmethod
  def process_op_weights(op_weights):
      while True:
          logits  = op_weights.log_softmax(dim=1)
          probs   = nn.functional.softmax(logits, dim=1)
          index   = probs.max(-1, keepdim=True)[1]
          one_h   = torch.zeros_like(logits).scatter_(-1, index, 1.0)
          hardwts = one_h - probs.detach() + probs
          if (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
              continue
          else: 
              break
      return hardwts, index
  def forward(self, inputs):
    
    if self._mode == "gdas":
        def get_gumbel_prob(xins):
            while True:
                gumbels = -torch.empty_like(xins).exponential_().log()
                logits  = (xins.log_softmax(dim=1) + gumbels) / self.tau
                probs   = nn.functional.softmax(logits, dim=1)
                index   = probs.max(-1, keepdim=True)[1]
                one_h   = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                    continue
                else: 
                    break
            return hardwts, index

        normal_hardwts, normal_index = get_gumbel_prob(self.arch_normal_parameters)
        reduce_hardwts, reduce_index = get_gumbel_prob(self.arch_reduce_parameters)

        s0 = s1 = self._stem(inputs)
        for i, cell in enumerate(self._cells):
            if cell.reduction: hardwts, index = reduce_hardwts, reduce_index
            else             : hardwts, index = normal_hardwts, normal_index
            s0, s1 = s1, cell.forward_gdas(s0, s1, hardwts, index)
        out = self.lastact(s1)
        out = self.global_pooling( out )
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        
    elif self.mode == "dynamic":
      # TODO this mirrors the GDAS branch more than it needs to
      normal_w, reduce_w = self.get_weights_from_arch((self.dynamic_cell.normal, self.dynamic_cell.reduce))
      normal_hardwts, normal_index = self.process_op_weights(normal_w)
      reduce_hardwts, reduce_index = self.process_op_weights(reduce_w)

      s0 = s1 = self._stem(inputs)
      for i, cell in enumerate(self._cells):
          if cell.reduction: hardwts, index = reduce_hardwts, reduce_index
          else             : hardwts, index = normal_hardwts, normal_index
          s0, s1 = s1, cell.forward_gdas(s0, s1, hardwts, index)
      out = self.lastact(s1)
      out = self.global_pooling( out )
      out = out.view(out.size(0), -1)
      logits = self.classifier(out)

    elif self._mode == "joint":
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
        
    elif self._mode == "urs":
      arch = self.random_topology_func()
      normal_w, reduce_w = self.get_weights_from_arch((arch.normal, arch.reduce))

      normal_hardwts, normal_index = self.process_op_weights(normal_w)
      reduce_hardwts, reduce_index = self.process_op_weights(reduce_w)

      s0 = s1 = self._stem(inputs)
      for i, cell in enumerate(self._cells):
          if cell.reduction: hardwts, index = reduce_hardwts, reduce_index
          else             : hardwts, index = normal_hardwts, normal_index
          s0, s1 = s1, cell.forward_gdas(s0, s1, hardwts, index)
      out = self.lastact(s1)
      out = self.global_pooling( out )
      out = out.view(out.size(0), -1)
      logits = self.classifier(out)
      pass
    
    else:
        print(f"Using mode={self.mode} which is not implemented")
        raise NotImplementedError

    if self.logits_only:
        return logits
    else:
        return out, logits

  def set_algo(self, algo: Text):
    # used for searching
    assert self._algo is None, 'This functioin can only be called once.'
    self._algo = algo
    # if algo == 'enas':
    #   self.controller = Controller(self.edge2index, self._op_names, self._max_nodes)
    # else:
    #   self.arch_parameters = nn.Parameter( 1e-3*torch.randn(self._num_edge, len(self._op_names)) )
    if algo == 'gdas':
        self._tau         = 10

  def set_drop_path(self, progress, drop_path_rate):
    pass
  def arch_params(self):
    return self.alphas
  @property
  def mode(self):
    return self._mode
  def set_tau(self, tau):
    self._tau = tau

  @property
  def tau(self):
    return self._tau
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
      
  def return_topK(self, K, use_random=False, size_percentile=None, perf_percentile=None, api=None, dataset=None):
    """NOTE additionaly outputs perf/size_all_dict.pkl mainly with shape {arch_str: perf_metric} """
    if use_random:
      sampled = self.arch_sampler.sample(mode="random", candidate_num=K)
      return sampled
    else:
      raise NotImplementedError

  def normalize_archp(self):
 
    alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)
    index   = alphas.max(-1, keepdim=True)[1]
    with torch.no_grad():
      alphas_cpu = alphas.detach().cpu()
    return alphas, alphas_cpu, index, 'SOFTMAX'

  def get_weights_from_arch(self, arch):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    # num_ops = len(genotypes.PRIMITIVES)
    num_ops = len(self._op_names)
    # n_nodes = self.model._steps
    n_nodes = self._max_nodes

    alphas_normal = nn.Parameter(torch.zeros(k, num_ops).cuda(), requires_grad=False)
    alphas_reduce = nn.Parameter(torch.zeros(k, num_ops).cuda(), requires_grad=False)

    offset = 0
    for i in range(n_nodes):
        normal1 = arch[0][2*i]
        normal2 = arch[0][2*i+1]
        reduce1 = arch[1][2*i]
        reduce2 = arch[1][2*i+1]

        alphas_normal[offset+normal1[1], DARTS_OPS_STR2IDX[normal1[0]]] = 1
        alphas_normal[offset+normal2[1], DARTS_OPS_STR2IDX[normal2[0]]] = 1
        alphas_reduce[offset+reduce1[1], DARTS_OPS_STR2IDX[reduce1[0]]] = 1
        alphas_reduce[offset+reduce2[1], DARTS_OPS_STR2IDX[reduce2[0]]] = 1
        offset += (i+2)

    arch_parameters = [
      alphas_normal,
      alphas_reduce,
    ]
    return arch_parameters