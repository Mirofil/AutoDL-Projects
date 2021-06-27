import numpy as np
import os
import sys
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from nasbench import api
from models.cell_searchs.nb101.nasbench_analysis.eval_random_search_ws_in_nasbench import eval_random_ws_model

def get_torch_home():
    if "TORCH_HOME" in os.environ:
        return os.environ["TORCH_HOME"]
    elif "HOME" in os.environ:
        return os.path.join(os.environ["HOME"], ".torch")
    else:
        raise ValueError(
            "Did not find HOME in os.environ. "
            "Please at least setup the path of HOME or TORCH_HOME "
            "in the environment."
        )
class IdentityDefaultDict(dict):
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, k):
        return k

class NASBench101Wrapper():
    def __init__(self, xargs) -> None:
        import nasbench
        from models.cell_searchs.nb101.nasbench_analysis.utils import NasbenchWrapper
        import os
        from collections import namedtuple
        
        try:
            api = NasbenchWrapper(os.path.join(get_torch_home() ,'nasbench_only108.tfrecord'))
        except:
            api = NasbenchWrapper(os.path.join(get_torch_home() ,'nasbench_full.tfrecord'))
        
        self.performance_model = api
        # self.archstr2index = defaultdict(lambda: "NB301 does not support this")
        self.archstr2index = IdentityDefaultDict()
        self.archs = {}
        self.xargs = xargs
        
    def query_index_by_arch(self, arch, **kwargs):
        return arch
    
    def query_str_by_arch(self, arch, hp = None):
        return "NASBench 301 does not really have this"
    
    def query_by_arch(self, arch, hp=None, is_random=False, **kwargs):

        return self.get_more_info(arch, is_random=is_random)
    
    def get_more_info(self, index, dataset=None, iepoch=None, hp=None, is_random=False, **kwargs):
        try:
            # adjacency_matrix, node_list = index[0], index[1]
            # adjacency_list = adjacency_matrix.astype(np.int).tolist()
            # model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
            # test_acc, valid_acc, runtime, params = self.performance_model.query(model_spec)
            
            test_acc, valid_acc = eval_random_ws_model({"search_space": int(self.xargs.search_space_paper.split("_")[1])}, model=index, nasbench=self.performance_model, from_file=False)
        except Exception as e:
            print(f"INDEX: {index}")
            # print(f"Adj matrix: {adjacency_matrix}, adj list: {adjacency_list}")
            print(f"Failed to get_more_info due to {e} with index={index}. Most likely the randomly sampled DARTS architecture is invalid")
            test_acc = -5
            valid_acc = -5
        results = {"train-loss" : 10, "train-accuracy": 10, "train-per-time":10000, "train-all-time": 10000,
                   "valid-loss": 10, "valid-accuracy": valid_acc, "valid-per-time": 10000, "valid-all-time": 10000,
                   "test-loss": 10, "test-accuracy": test_acc, "test-per-time": 10000, "test-all-time": 10000,
                   "valtest-loss": 10, "valtest-accuracy": 10, "valtest-per-time": 10000, "valtest-all-time": 10000}
        return results
        
    def get_cost_info(self, index, dataset=None, hp=None):
        return {"flops":10, "params": 10, "latency":10, "NOT SUPPORTED FOR NB301": None}