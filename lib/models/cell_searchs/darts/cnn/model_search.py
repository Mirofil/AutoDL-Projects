import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
import genotypes
from genotypes import PRIMITIVES, PRIMITIVES_STR2IDX
# from genotypes import Genotype
from typing import *
from copy import deepcopy
import numpy as np
import copy
from collections import namedtuple
import random

Genotype_tuple = namedtuple('Genotype_tuple', 'normal normal_concat reduce reduce_concat')

class Genotype:
  def __init__(self, normal, normal_concat, reduce, reduce_concat) -> None:
      self.normal = normal
      self.normal_concat = normal_concat
      self.reduce = reduce
      self.reduce_concat = reduce_concat
      self.genotype_tuple = Genotype_tuple(normal, normal_concat, reduce, reduce_concat)
      self.arch_tuple = (normal, reduce)
      
  def tostr(self):
    return str(self.genotype_tuple)

  def __hash__(self):
    return hash(str(self.genotype_tuple))
    
  def __repr__(self):
    return str(self.genotype_tuple)
  
  def __getitem__(self, k):
    if type(k) is str:
      return getattr(self, k)
    elif type(k) is int:
      return self.arch_tuple[k]
    else:
      raise NotImplementedError
      
  

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) if w != 0.0 else 0 for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, discrete=False):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        if discrete:
          weights = self.alphas_reduce
        else:
          weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        if discrete:
          weights = self.alphas_normal
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters
  def arch_params(self):
    return self.arch_parameters()
  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype


class NetworkNB(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, discrete=True):
    super(NetworkNB, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._op_names = PRIMITIVES

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
    self._stem = self.stem
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr


    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self._cells = self.cells
    
    self._initialize_alphas()
    # NEWLY ADDED STUFF
    self._mode        = None
    self.dynamic_cell = None
    self._tau         = None
    self._algo        = None
    self._drop_path   = None
    self.verbose      = False
    self.logits_only = False
    self.arch_sampler = None
    self.discrete = discrete
    self._max_nodes = 4 # TODO should be the same I think?
    print(f"Instantiated DARTS model from RandomNAS with discrete={discrete}")
    
    
  def new(self):
    model_new = NetworkNB(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, discrete=None):
    if discrete is None:
      discrete = self.discrete
    
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      # normal_w, reduce_w = self.get_weights_from_arch((self.dynamic_cell.normal, self.dynamic_cell.reduce))
      if cell.reduction:
        
        if discrete:
          weights = self.arch_reduce_parameters
        elif self._mode in ["dynamic", "urs", "gdas", "enas"]:
          weights = self.dynamic_cell.reduce
        else:
          weights = F.softmax(self.arch_reduce_parameters, dim=-1)
      else:
        if discrete:
          weights = self.arch_normal_parameters
        elif self._mode in ["dynamic", "urs", "gdas", "enas"]:
          weights = self.dynamic_cell.normal
        else:
          weights = F.softmax(self.arch_normal_parameters, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    if self.logits_only:
      return logits
    else:
      return "Placeholder to make it match Nasbench 201", logits

  def forward_old(self, input, discrete=False):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        if discrete:
          weights = self.arch_reduce_parameters
        else:
          weights = F.softmax(self.arch_reduce_parameters, dim=-1)
      else:
        if discrete:
          weights = self.arch_normal_parameters
        else:
          weights = F.softmax(self.arch_normal_parameters, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.arch_normal_parameters = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.arch_reduce_parameters = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    
    self._arch_parameters = [
      self.arch_normal_parameters,
      self.arch_reduce_parameters,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.arch_normal_parameters, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.arch_reduce_parameters, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def set_algo(self, algo):
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

    sampled = [self.random_topology_func(nb301_format=True) for _ in range(K)]
    return sampled

  def normalize_archp(self):

    alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)
    index   = alphas.max(-1, keepdim=True)[1]
    with torch.no_grad():
      alphas_cpu = alphas.detach().cpu()
    return alphas, alphas_cpu, index, 'SOFTMAX'

  def _parse(self,weights, original_darts_format=False):
    gene = []
    n = 2
    start = 0
    for i in range(self._steps):
      end = start + n
      W = weights[start:end].copy()
      edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if ('none' not in PRIMITIVES or k != PRIMITIVES.index('none'))))[:2]
      for j in edges:
        k_best = None
        for k in range(len(W[j])):
          if ('none' not in PRIMITIVES) or (k != PRIMITIVES.index('none')):
            if k_best is None or W[j][k] > W[j][k_best]:
              k_best = k
        gene.append((PRIMITIVES[k_best], j))
      start = end
      n += 1
    return gene

  # def _parse(self, weights, original_darts_format=False):
  #   gene = []
  #   for i in range(self._steps):
  #     edges = []
  #     for j in range(2+i):
  #       node_str = '{:}<-{:}'.format(i, j)
  #       ws = weights[ self.edge2index[node_str] ]
  #       for k, op_name in enumerate(self._op_names):
  #         if op_name == 'none': continue
  #         if not original_darts_format:
  #           edges.append( (op_name, j, ws[k]) )
  #         else:
  #           edges.append((op_name, j, ws[k]))
  #     # (TODO) xuanyidong:
  #     # Here the selected two edges might come from the same input node.
  #     # And this case could be a problem that two edges will collapse into a single one
  #     # due to our assumption -- at most one edge from an input node during evaluation.
  #     edges = sorted(edges, key=lambda x: -x[-1]) # NOTE This is the argmax over softmax coefficients part
  #     selected_edges = edges[:2]
  #     if not original_darts_format:
  #       gene.append( tuple(selected_edges) )
  #     else:
  #       gene.extend([(x[0], x[1]) for x in selected_edges])
  #   return gene

  def set_cal_mode(self, mode, dynamic_cell=None, sandwich_cells=None):
    assert mode in ['gdas', 'enas', 'urs', 'joint', 'select', 'dynamic', 'sandwich']
    self._mode = mode
    if mode == 'dynamic':
      # self.dynamic_cell = deepcopy(dynamic_cell)
      weights = self.get_weights_from_arch(dynamic_cell)
      self.set_model_weights(weights)
      
    else                : self.dynamic_cell = None
    if mode == "sandwich":
      assert sandwich_cells is not None
      self.sandwich_cells = sandwich_cells
      
    # if mode in ['gdas', 'urs', 'select', 'dynamic', 'sandwich']:
    #   self.discrete = True
    # elif mode in ['joint']:
    #   self.discrete = False
      
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
  def get_genotype(self, original_darts_format=True) -> Dict[Text, List]:
    # Used for NASBench 301 purposes among other things
    with torch.no_grad():
      gene_normal = self._parse(torch.softmax(torch.randn_like(self.arch_normal_parameters), dim=-1).cpu().numpy(), original_darts_format=original_darts_format)
      gene_reduce = self._parse(torch.softmax(torch.randn_like(self.arch_reduce_parameters), dim=-1).cpu().numpy(), original_darts_format=original_darts_format)
    return Genotype(normal = gene_normal, normal_concat = list(range(2+self._steps-self._multiplier, self._steps+2)), 
                reduce = gene_reduce, reduce_concat = list(range(2+self._steps-self._multiplier, self._steps+2)))
    # return {'normal': gene_normal, 'normal_concat': list(range(2+self._steps-self._multiplier, self._steps+2)),
    #         'reduce': gene_reduce, 'reduce_concat': list(range(2+self._steps-self._multiplier, self._steps+2))}
  def random_topology_func(self, k=1, nb301_format=False):
    # NOTE Here it is mainly as compatibility layer with previous NB201 code 
    
    # k = sum(1 for i in range(self._steps) for n in range(2+i))
    # # num_ops = len(genotypes.PRIMITIVES)
    # num_ops = len(self._op_names)
    # # n_nodes = self._steps
    # n_nodes = self._max_nodes

    # normal = []
    # reduction = []
    # for i in range(n_nodes):
    #     ops = np.random.choice(range(num_ops), 4)
    #     nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
    #     nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
    #     normal.extend([(PRIMITIVES[ops[0]], nodes_in_normal[0]), (PRIMITIVES[ops[1]], nodes_in_normal[1])])
    #     reduction.extend([(PRIMITIVES[ops[2]], nodes_in_reduce[0]), (PRIMITIVES[ops[3]], nodes_in_reduce[1])])
    # # return (normal, reduction)
    
    archs = [self.sample_arch(nb301_format=nb301_format) for _ in range(k)]
    
    result = [Genotype(normal = arch[0], normal_concat = list(range(2+self._steps-self._multiplier, self._steps+2)), 
                reduce = arch[1], reduce_concat = list(range(2+self._steps-self._multiplier, self._steps+2))) for arch in archs]
    if k == 1:
      return result[0]
    else:
      return result

  def convert_tuple_to_genotype(self, gene_tuple):
    return Genotype(normal=gene_tuple[0], normal_concat = list(range(2+self._steps-self._multiplier, self._steps+2)),
                    reduce_concat = list(range(2+self._steps-self._multiplier, self._steps+2)), reduce=gene_tuple[1])

  def get_weights(self) -> List[torch.nn.Parameter]:
    xlist = list( self._stem.parameters() ) + list( self._cells.parameters() )
    xlist+= list( self.global_pooling.parameters() )
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
    return ("NetworkNB model (TODO get better repr)")


  ### THIS PART IS FROM RANDOM_NAS
  def get_weights_from_arch(self, arch):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = self._steps

    alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
    alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

    offset = 0
    for i in range(n_nodes):
        normal1 = arch[0][2*i] 
        normal2 = arch[0][2*i+1]
        reduce1 = arch[1][2*i]
        reduce2 = arch[1][2*i+1]
        # print(type(normal1[0]))
        offset_normal1 = (normal1[0] if isinstance(normal1[0], (int, np.int32)) else PRIMITIVES_STR2IDX[normal1[0]])
        offset_normal2 = (normal2[0] if isinstance(normal2[0], (int, np.int32)) else PRIMITIVES_STR2IDX[normal2[0]])
        offset_reduce1 = (reduce1[0] if isinstance(reduce1[0], (int, np.int32)) else PRIMITIVES_STR2IDX[reduce1[0]])
        offset_reduce2 = (reduce2[0] if isinstance(reduce2[0], (int, np.int32)) else PRIMITIVES_STR2IDX[reduce2[0]])
        alphas_normal[offset+normal1[1], offset_normal1] = 1
        alphas_normal[offset+normal1[1], offset_normal2] = 1
        alphas_reduce[offset+reduce1[1], offset_reduce1] = 1
        alphas_reduce[offset+reduce2[1], offset_reduce2] = 1
        offset += (i+2)

    arch_parameters = [
      alphas_normal,
      alphas_reduce,
    ]
    return arch_parameters

  def set_model_weights(self, weights):
    # weights should be a tuple of (normal_weights, reduce_weights)
    if self.discrete:
      self.arch_normal_parameters = weights[0]
      self.arch_reduce_parameters = weights[1]
      self._arch_parameters = [self.arch_normal_parameters, self.arch_reduce_parameters]
    self.dynamic_cell = Genotype(normal=weights[0], reduce = weights[1], normal_concat=[2,3,4,5], reduce_concat=[2,3,4,5])

  def sample_arch(self, nb301_format=True):
      k = sum(1 for i in range(self._steps) for n in range(2+i))
      num_ops = len(genotypes.PRIMITIVES)
      n_nodes = self._steps

      normal = []
      reduction = []
      for i in range(n_nodes):
          ops = np.random.choice(range(num_ops), 4)
          nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
          nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
          if not nb301_format:
            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
          else:
            normal.extend([(PRIMITIVES[ops[0]], nodes_in_normal[0]), (PRIMITIVES[ops[1]], nodes_in_normal[1])])
            reduction.extend([(PRIMITIVES[ops[2]], nodes_in_reduce[0]), (PRIMITIVES[ops[3]], nodes_in_reduce[1])])

      return (normal, reduction)

  def sample_archs_fairnas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(genotypes.PRIMITIVES)
    n_nodes = self._steps

    normals = [[] for _ in range(num_ops)]
    reductions = [[] for _ in range(num_ops)]
    for i in range(n_nodes):
        ops_normal1 = random.sample(range(num_ops), num_ops)
        ops_normal2 = random.sample(range(num_ops), num_ops)

        ops_reduce1 = random.sample(range(num_ops), num_ops)
        ops_reduce2 = random.sample(range(num_ops), num_ops)

        nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
        nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
        for i in range(num_ops):
          normals[i].extend([(PRIMITIVES[ops_normal1[i]], nodes_in_normal[0]), (PRIMITIVES[ops_normal2[i]], nodes_in_normal[1])])
          reductions[i].extend([(PRIMITIVES[ops_reduce1[i]], nodes_in_reduce[0]), (PRIMITIVES[ops_reduce2[i]], nodes_in_reduce[1])])

    return [self.convert_tuple_to_genotype((normal, reduction)) for normal, reduction in zip(normals, reductions)]

  def perturb_arch(self, arch):
      new_arch = copy.deepcopy(arch)
      num_ops = len(genotypes.PRIMITIVES)

      cell_ind = np.random.choice(2)
      step_ind = np.random.choice(self._steps)
      nodes_in = np.random.choice(step_ind+2, 2, replace=False)
      ops = np.random.choice(range(num_ops), 2)

      new_arch[cell_ind][2*step_ind] = (nodes_in[0], ops[0])
      new_arch[cell_ind][2*step_ind+1] = (nodes_in[1], ops[1])
      return new_arch
