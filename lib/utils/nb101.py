from sotl_utils import get_torch_home


class IdentityDefaultDict(dict):
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, k):
        return k

class NASBench101Wrapper():
    def __init__(self) -> None:
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
        
    def query_index_by_arch(self, arch, **kwargs):
        return arch
    
    def query_str_by_arch(self, arch, hp = None):
        return "NASBench 301 does not really have this"
    
    def query_by_arch(self, arch, hp=None, is_random=False, **kwargs):
        return self.get_more_info(arch, is_random=is_random)
    
    def get_more_info(self, index, dataset=None, iepoch=None, hp=None, is_random=False, **kwargs):
        try:
            if type(index) is str:
                index = eval(index)
            test_acc, valid_acc, runtime, params = self.performance_model.query(index)
        except Exception as e:
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