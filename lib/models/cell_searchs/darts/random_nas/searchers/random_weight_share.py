# python ./lib/models/cell_searchs/darts/random_nas/searchers/random_weight_share.py

import sys
# sys.path.append('/home/liamli4465/nas_weight_share')
# sys.path.append('../')
from pathlib import Path
lib_dir = (Path(__file__).parent / '..').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

import os
import shutil
import logging
import inspect
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import scipy.stats
import wandb

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

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
        
def load_nb301():
    import nasbench301 as nb
    version = '0.9'
    current_dir = os.path.dirname(get_torch_home())
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

    # Load the performance surrogate model
    #NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
    #NOTE: Defaults to using the default model download path
    print("==> Loading performance surrogate model...")
    ensemble_dir_performance = model_paths['xgb']
    print(ensemble_dir_performance)
    performance_model = nb.load_ensemble(ensemble_dir_performance)
    
    return performance_model
class Rung:
    def __init__(self, rung, nodes):
        self.parents = set()
        self.children = set()
        self.rung = rung
        for node in nodes:
            n = nodes[node]
            if n.rung == self.rung:
                self.parents.add(n.parent)
                self.children.add(n.node_id)

class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung
    def to_dict(self):
        out = {'parent':self.parent, 'arch': self.arch, 'node_id': self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out

class Random_NAS:
    def __init__(self, B, model, seed, save_dir, args):
        self.save_dir = save_dir

        self.B = B
        self.model = model
        self.seed = seed

        self.iters = 0

        self.arms = {}
        self.node_id = 0
        self.api = load_nb301()
        self.args = args

    def print_summary(self):
        logging.info(self.parents)
        objective_vals = [(n,self.arms[n].objective_val) for n in self.arms if hasattr(self.arms[n],'objective_val')]
        objective_vals = sorted(objective_vals,key=lambda x:x[1])
        best_arm = self.arms[objective_vals[0][0]]
        val_ppl = self.model.evaluate(best_arm.arch, split='valid')
        logging.info(objective_vals)
        logging.info('best valid ppl: %.2f' % val_ppl)


    def get_arch(self):
        arch, arch_nb = self.model.sample_arch()
        self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)
        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last pickle save
        with open(os.path.join(self.save_dir,'results_tmp.pkl'),'wb') as f:
            pickle.dump(to_save, f)
        shutil.copyfile(os.path.join(self.save_dir, 'results_tmp.pkl'), os.path.join(self.save_dir, 'results.pkl'))

        self.model.save()

    def run(self):
        while self.iters < self.B:
            # if self.iters > 10:
            #     break
            arch = self.get_arch()
            self.model.train_batch([arch])
            self.iters += 1
            if self.iters % 500 == 0:
                self.save()
        self.save()

    def get_eval_arch(self, rounds=None):
        #n_rounds = int(self.B / 7 / 1000)
        if rounds is None:
            n_rounds = max(1,int(self.B/10000))
        else:
            n_rounds = rounds
        best_rounds = []
        for r in range(n_rounds):
            sample_vals = {"tl_mini":{}, "vl_mini":{}, "sotl":{}, "valacc_mini":{}, "valacc_total":{}, "vl_total":{}}
            archs, archs_nb = [], []
            for arch_idx in tqdm(range(self.args.eval_candidate_num), desc = "Iterating over archs"):
                arch, arch_nb = self.model.sample_arch()
                archs.append(arch)
                archs_nb.append(arch_nb)
                for k in sample_vals.keys():
                    sample_vals[k][str(arch)] = []
                
                train_losses_list, valid_loss_list, valid_acc_list, valid_loss_mini_list, valid_acc_mini_list = self.model.evaluate(arch, eval_metric="sotltrain", verbose=True if arch_idx < 3 else False)
                logging.info(arch_nb)
                # logging.info('objective_val: %.3f' % ppl)
                # sample_vals.append((arch, ppl))
                for n in [0, 100, 200, 300]:
                    if n >= len(train_losses_list):
                        break
                    sample_vals["tl_mini"][str(arch)].append((str(arch_nb), train_losses_list[n]))
                    sample_vals["vl_mini"][str(arch)].append((str(arch_nb), valid_loss_mini_list[n]))
                    sample_vals["valacc_mini"][str(arch)].append((str(arch_nb), valid_acc_mini_list[n]))
                    sample_vals["sotl"][str(arch)].append((str(arch_nb), sum(train_losses_list[0:n+1])))
                    sample_vals["valacc_total"][str(arch)].append((str(arch_nb), valid_acc_list[n]))
                    sample_vals["vl_total"][str(arch)].append((str(arch_nb), valid_loss_list[n]))

            # for k in sample_vals.keys():
            #     sample_vals[k] = sorted(sample_vals[k], key=lambda x:str(x[0])) # Sort by the architecture hash! So we can easily compute correlatiosn later
            true_perfs = [(str(arch_nb),  self.api.predict(config=Genotype(normal=arch_nb[0], reduce=arch_nb[1], normal_concat=range(2,6), reduce_concat=range(2,6)), 
                                              representation='genotype', with_noise=False)) for arch_nb in archs_nb]
            true_perfs = sorted(true_perfs, key=lambda x: str(x[0])) # Sort by architecture hash again
            
            for n in [0, 100, 200, 300]:
                if n >= len(train_losses_list):
                        break
                for k in ["sotl", "tl_mini", "vl_mini", "valacc_mini", "vl_total", "valacc_total"]:
                    reshaped = sorted([(arch, sample_vals[k][str(arch)][int(n/100)][1]) for arch in archs], key = lambda x: str(x[0]))
                    reshaped = [x[1] for x in reshaped]
                    corr = scipy.stats.spearmanr(reshaped, [x[1] for x in true_perfs]).correlation
                    print(f"Corr for {k} at n={n} is {corr}")
                    
            # if 'split' in inspect.getargspec(self.model.evaluate).args:
            #     for i in range(10):
            #         arch = sample_vals[i][0]
            #         try:
            #             ppl = self.model.evaluate(arch, split='valid')
            #         except Exception as e:
            #             ppl = 1000000
            #         full_vals.append((arch, ppl))
            #     full_vals = sorted(full_vals, key=lambda x:x[1])
            #     logging.info('best arch: %s, best arch valid performance: %.3f' % (' '.join([str(i) for i in full_vals[0][0]]), full_vals[0][1]))
            #     best_rounds.append(full_vals[0])
            # else:
            #     best_rounds.append(sample_vals[0])
            
        best_rounds = []
        return best_rounds
    
def wandb_auth(fname: str = "nas_key.txt"):
  gdrive_path = "/content/drive/MyDrive/colab/wandb/nas_key.txt"
  if "WANDB_API_KEY" in os.environ:
      wandb_key = os.environ["WANDB_API_KEY"]
  elif os.path.exists(os.path.abspath("~" + os.sep + ".wandb" + os.sep + fname)):
      # This branch does not seem to work as expected on Paperspace - it gives '/storage/~/.wandb/nas_key.txt'
      print("Retrieving WANDB key from file")
      f = open("~" + os.sep + ".wandb" + os.sep + fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists("/root/.wandb/"+fname):
      print("Retrieving WANDB key from file")
      f = open("/root/.wandb/"+fname, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key

  elif os.path.exists(
      os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname
  ):
      print("Retrieving WANDB key from file")
      f = open(
          os.path.expandvars("%userprofile%") + os.sep + ".wandb" + os.sep + fname,
          "r",
      )
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  elif os.path.exists(gdrive_path):
      print("Retrieving WANDB key from file")
      f = open(gdrive_path, "r")
      key = f.read().strip()
      os.environ["WANDB_API_KEY"] = key
  wandb.login()
  
def main(args):
    # Fill in with root output path
    root_dir = './checkpoints/'
    if args.save_dir is None:
        save_dir = os.path.join(root_dir, '%s/random/trial%d' % (args.benchmark, args.seed))
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if args.eval_only:
    #     assert args.save_dir is not None
        
        
    wandb_auth()
    run = wandb.init(project="NAS", group=f"Search_Cell_darts_orig", reinit=True)


    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(args)

    if args.benchmark=='ptb':
        data_size = 929589
        time_steps = 35
    else:
        data_size = 25000
        time_steps = 1
    B = int(args.epochs * data_size / args.batch_size / time_steps)
    if args.benchmark=='ptb':
        from benchmarks.ptb.darts.darts_wrapper_discrete import DartsWrapper
        model = DartsWrapper(save_dir, args.seed, args.batch_size, args.grad_clip, config=args.config)
    elif args.benchmark=='cnn':
        from benchmarks.cnn.darts.darts_wrapper_discrete_fair import DartsWrapper
        model = DartsWrapper(save_dir, args.seed, args.batch_size, args.grad_clip, args.epochs, init_channels=args.init_channels, 
                             finetune_lr=args.finetune_lr, consistent_finetune_order=args.consistent_finetune_order)

    searcher = Random_NAS(B, model, args.seed, save_dir, args=args)
    logging.info('budget: %d' % (searcher.B))
    if not args.eval_only:
        searcher.run()
        archs = searcher.get_eval_arch(1)
    else:
        np.random.seed(args.seed+1)
        archs = searcher.get_eval_arch(1)
    logging.info(archs)
    # arch = ' '.join([str(a) for a in archs[0][0]])
    # with open('/tmp/arch','w') as f:
    #     f.write(arch)
    arch = []
    return arch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
    parser.add_argument('--benchmark', dest='benchmark', type=str, default='cnn')
    parser.add_argument('--seed', dest='seed', type=int, default=100)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.25)
    parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
    parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
    # PTB only argument. config=search uses proxy network for shared weights while
    # config=eval uses proxyless network for shared weights.
    parser.add_argument('--config', dest='config', type=str, default="search")
    # CIFAR-10 only argument.  Use either 16 or 24 for the settings for random search
    # with weight-sharing used in our experiments.
    parser.add_argument('--init_channels', dest='init_channels', type=int, default=16)
    parser.add_argument('--eval_candidate_num', dest='eval_candidate_num', type=int, default=100)
    parser.add_argument('--finetune_lr', dest='finetune_lr', type=float, default=0.001)
    parser.add_argument('--consistent_finetune_order', dest='consistent_finetune_order', type=lambda x: False if x in ["False", "false", "", "None", False, None] else True,
                        default=False)

    
    args = parser.parse_args()

    main(args)







