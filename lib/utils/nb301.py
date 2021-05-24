import os

class NASBench301Wrapper():
    def __init__(self) -> None:
        import nasbench301
        import os
        from collections import namedtuple

        from ConfigSpace.read_and_write import json as cs_json
        from collections import defaultdict
        import nasbench301 as nb
        version = '0.9'

        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
        model_paths_0_9 = {
            model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
            for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
        }
        models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
        model_paths_1_0 = {
            model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
            for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
        }
        model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

        # If the models are not available at the paths, automatically download
        # the models
        # Note: If you would like to provide your own model locations, comment this out
        if not all(os.path.exists(model) for model in model_paths.values()):
            nb.download_models(version=version, delete_zip=True,
                            download_dir=current_dir)
        ensemble_dir_performance = model_paths['xgb']
        performance_model = nb.load_ensemble(ensemble_dir_performance)
        
        self.performance_model = performance_model
        self.archstr2index = defaultdict(lambda: "NB301 does not support this")
        self.archs = {}
        
    def query_index_by_arch(self, arch, **kwargs):
        return arch
    
    def query_str_by_arch(self, arch, hp = None):
        return "NASBench 301 does not really have this"
    
    def get_more_info(self, index, dataset=None, iepoch=None, hp=None, is_random=False, **kwargs):
        true_acc = self.performance_model.predict(config=index, representation="genotype", with_noise=is_random)
        results = {"train-loss" : 10, "train-accuracy": 10, "train-per-time":10000, "train-all-time": 10000,
                   "valid-loss": 10, "valid-accuracy": true_acc, "valid-per-time": 10000, "valid-all-time": 10000,
                   "test-loss": 10, "test-accuracy": true_acc, "test-per-time": 10000, "test-all-time": 10000,
                   "valtest-loss": 10, "valtest-accuracy": 10, "valtest-per-time": 10000, "valtest-all-time": 10000}
        return results
        
    def get_cost_info(self, index, dataset=None, hp=None):
        return {"flops":10, "params": 10, "latency":10, "NOT SUPPORTED FOR NB301": None}
    

def download_natsbench(output):
  import gdown
  import tarfile
  output="/root/.torch/nats-bench.tar"
  url = 'https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU'
  gdown.download(url, output, quiet=False)
  my_tar = tarfile.open(output)
  my_tar.extract_all()