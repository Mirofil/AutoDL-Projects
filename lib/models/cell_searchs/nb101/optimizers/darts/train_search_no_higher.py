import argparse
import glob
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

lib_dir = (Path(__file__).parent / '..' / '..' / '..' / '..'/ '..' / '..' /'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))


from nasbench_analysis import eval_darts_one_shot_model_in_nasbench as naseval
from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from optimizers.darts import utils
from optimizers.darts.architect import Architect
from optimizers.darts.model_search import Network

from optimizers.sotl_utils import wandb_auth
import wandb
from pathlib import Path
from tqdm import tqdm
from utils.train_loop import approx_hessian, exact_hessian
from datasets     import get_datasets, get_nas_search_loaders

from nasbench import api
from copy import deepcopy
from nasbench_analysis.utils import NasbenchWrapper

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the darts corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=9, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random_ws seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training darts')
parser.add_argument('--unrolled',type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--output_weights', type=bool, default=True, help='Whether to use weights on the output nodes')
parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
parser.add_argument('--debug', action='store_true', default=False, help='run only for some batches')
parser.add_argument('--warm_start_epochs', type=int, default=0,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--steps_per_epoch', type=float, default=None, help='weight decay for arch encoding')
parser.add_argument('--inner_steps', type=int, default=100, help='Steps for inner loop of bilevel')
parser.add_argument('--bilevel_train_steps', type=int, default=None, help='Steps for inner loop of bilevel')

parser.add_argument('--higher_method' ,       type=str, choices=['val', 'sotl', "val_multiple", "sotl_v2"],   default='sotl', help='Whether to take meta gradients with respect to SoTL or val set (which might be the same as training set if they were merged)')
parser.add_argument('--merge_train_val', type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=False, help='portion of training data')
parser.add_argument('--perturb_alpha', type=str, default=None, help='portion of training data')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')

parser.add_argument('--hessian', type=lambda x: False if x in ["False", "false", "", "None", False, None] else True, default=True,
                    help='Warm start one-shot model before starting architecture updates.')
parser.add_argument('--dataset', type=str, default="cifar10",
                    help='Warm start one-shot model before starting architecture updates.')

parser.add_argument('--total_samples',          type=int, default=None, help='Number of total samples in dataset. Useful for limiting Cifar5m')
parser.add_argument('--data_path'   ,       type=str,default="$TORCH_HOME/cifar.python",    help='Path to dataset')
parser.add_argument('--mmap',          type=str, default="r", help='Whether to mmap cifar5m')

parser.add_argument('--mode' ,       type=str,   default="higher", choices=["higher", "reptile"], help='Number of steps to do in the inner loop of bilevel meta-learning')

args = parser.parse_args()

args.save = 'experiments/darts/search_space_{}/search-no_higher-{}-{}-{}-{}'.format(args.search_space, args.save,
                                                                          time.strftime("%Y%m%d-%H%M%S"), args.seed,
                                                                          args.search_space)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# Dump the config of the run
with open(os.path.join(args.save, 'config.json'), 'w') as fp:
    json.dump(args.__dict__, fp)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()

CIFAR_CLASSES = 10

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

def main():
    # Select the search space to search in
    if args.search_space == '1':
        search_space = SearchSpace1()
    elif args.search_space == '2':
        search_space = SearchSpace2()
    elif args.search_space == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    logger = logging.getLogger()

    wandb_auth()
    run = wandb.init(project="NAS", group=f"Search_Cell_nb101", reinit=True)
    wandb.config.update(args)
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, output_weights=args.output_weights,
                    steps=search_space.num_intermediate_nodes, search_space=search_space)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.weights_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        if args.dataset == "cifar10":
          train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        elif args.dataset == "cifar100":
          train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True)

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True)
        
    elif args.dataset == "cifar5m":
        train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_path, -1, mmap=args.mmap, total_samples=args.total_samples)
        _, train_queue, valid_queue = get_nas_search_loaders(train_data, valid_data, args.dataset, 'configs/nas-benchmark/', 
            (args.batch_size, args.batch_size), workers=0, 
            epochs=args.epochs, determinism="all", 
            merge_train_val = False, merge_train_val_and_use_test = False, 
            extra_split = True, valid_ratio=1, use_only_train=True, xargs=args)
        train_queue.sampler.auto_counter = True
        valid_queue.sampler.auto_counter = True
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    if args.merge_train_val:
        valid_queue = train_queue
    architect = Architect(model, args)
    
    # if os.path.exists(Path(args.save) / "checkpoint.pt"):
    #     checkpoint = torch.load(Path(args.save) / "checkpoint.pt")
    #     optimizer.load_state_dict(checkpoint["w_optimizer"])
    #     architect.optimizer.load_state_dict(checkpoint["a_optimizer"])
    #     model.load_state_dict(checkpoint["model"])
    #     scheduler.load_state_dict(checkpoint["w_scheduler"])
    #     start_epoch = checkpoint["epoch"]
    #     all_logs = checkpoint["all_logs"]

    # else:
    #     print(f"Path at {Path(args.save) / 'checkpoint.pt'} does not exist")
    #     start_epoch=0
    #     all_logs=[]
    all_logs = []
    try:
        nasbench = NasbenchWrapper(os.path.join(get_torch_home() ,'nasbench_only108.tfrecord'))

    except:
        nasbench = NasbenchWrapper(os.path.join(get_torch_home() ,'nasbench_full.tfrecord'))

    for epoch in tqdm(range(args.epochs), desc = "Iterating over epochs"):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        # increase the cutout probability linearly throughout search
        train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
        logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                     train_transform.transforms[-1].cutout_prob)

        # Save the one shot model architecture weights for later analysis
        arch_filename = os.path.join(args.save, 'one_shot_architecture_{}.obj'.format(epoch))
        with open(arch_filename, 'wb') as filehandler:
            numpy_tensor_list = []
            for tensor in model.arch_parameters():
                numpy_tensor_list.append(tensor.detach().cpu().numpy())
            pickle.dump(numpy_tensor_list, filehandler)

        # Save the entire one-shot-model
        filepath = os.path.join(args.save, 'one_shot_model_{}.obj'.format(epoch))
        torch.save(model.state_dict(), filepath)

        logging.info(f'architecture : {numpy_tensor_list}')
        
        if args.perturb_alpha:
          epsilon_alpha = 0.03 + (args.epsilon_alpha - 0.03) * epoch / args.epochs
          logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)
        else:
          epsilon_alpha = None
            
        # training
        if args.mode == "higher":
            train_acc, train_obj = train(train_queue=train_queue, valid_queue=valid_queue, network=model, architect=architect, 
                                        criterion=criterion, w_optimizer=optimizer, a_optimizer=architect.optimizer,
                                        logger=logger, inner_steps=args.inner_steps, epoch=epoch, steps_per_epoch=args.steps_per_epoch, epsilon_alpha=epsilon_alpha,
                                        perturb_alpha=utils.Random_alpha)
        elif args.mode == "reptile":
            train_acc, train_obj = train_reptile(train_queue=train_queue, valid_queue=valid_queue, network=model, architect=architect, 
                                criterion=criterion, w_optimizer=optimizer, a_optimizer=architect.optimizer,
                                logger=logger, inner_steps=args.inner_steps, epoch=epoch, steps_per_epoch=args.steps_per_epoch, epsilon_alpha=epsilon_alpha,
                                perturb_alpha=utils.Random_alpha)

        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        
        genotype_perf, _, _, _ = naseval.eval_one_shot_model(config=args.__dict__,
                                                               model=arch_filename, nasbench=nasbench)
        print(f"Genotype performance: {genotype_perf}" )
        if args.hessian and torch.cuda.get_device_properties(0).total_memory < 15147483648:
            eigenvalues = approx_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, args=args)
            # eigenvalues = exact_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, epoch=epoch, logger=logger, args=args)
        elif False and args.hessian and torch.cuda.get_device_properties(0).total_memory > 15147483648:
            eigenvalues = exact_hessian(network=model, val_loader=valid_queue, criterion=criterion, xloader=valid_queue, epoch=epoch, logger=logger, args=args)

        else:
            eigenvalues = None
        
        
        wandb_log = {"train_acc":train_acc, "train_loss":train_obj, "val_acc": valid_acc, "valid_loss":valid_obj,
                     "search.final.cifar10": genotype_perf, "epoch":epoch, "eigval":eigenvalues}
        all_logs.append(wandb_log)
        wandb.log(wandb_log)
        
        utils.save_checkpoint2({"model":model.state_dict(), "w_optimizer":optimizer.state_dict(), 
                           "a_optimizer":architect.optimizer.state_dict(), "w_scheduler":scheduler.state_dict(), "epoch": epoch, 
                           "all_logs":all_logs}, 
                          Path(args.save) / "checkpoint.pt")
        # utils.save(model, os.path.join(args.save, 'weights.pt'))

    logging.info('STARTING EVALUATION')
    test, valid, runtime, params = naseval.eval_one_shot_model(config=args.__dict__,
                                                               model=arch_filename, nasbench=nasbench)
    index = 0
    logging.info('TEST ERROR: %.3f | VALID ERROR: %.3f | RUNTIME: %f | PARAMS: %d'
                 % (test,
                    valid,
                    runtime,
                    params)
                 )
    wandb.log({"test_error":test, "valid_error": valid, "runtime":runtime, "params":params})
    for log in tqdm(all_logs, desc = "Logging search logs"):
        wandb.log(log)


def train(train_queue, valid_queue, network, architect, criterion, w_optimizer, a_optimizer, logger=None, inner_steps=100, epoch=0, 
          steps_per_epoch=None, perturb_alpha=None, epsilon_alpha=None):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    train_iter = iter(train_queue)
    valid_iter = iter(valid_queue)
    search_loader_iter = zip(train_iter, valid_iter)
    for data_step, ((base_inputs, base_targets), (arch_inputs, arch_targets)) in tqdm(enumerate(search_loader_iter), total = round(len(train_queue)/inner_steps)):
      if steps_per_epoch is not None and data_step > steps_per_epoch:
        break
      network.train()
      n = base_inputs.size(0)

      base_inputs = base_inputs.cuda()
      base_targets = base_targets.cuda(non_blocking=True)

      # get a random minibatch from the search queue with replacement
      input_search, target_search = next(iter(valid_queue))
      input_search = input_search.cuda()
      target_search = target_search.cuda(non_blocking=True)
      
      all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, 
                                                                                                search_loader_iter, inner_steps=inner_steps, args=args)

      network.zero_grad()

      model_init = deepcopy(network.state_dict())
      w_optim_init = deepcopy(w_optimizer.state_dict())
      arch_grads = [torch.zeros_like(p) for p in network.arch_parameters()]
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in tqdm(enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)), desc = "Unrolling bilevel loop", total=inner_steps, disable=True):
          if data_step < 2 and inner_step < 2:
              print(f"Base targets in the inner loop at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
            
          logits = network(base_inputs)
          base_loss = criterion(logits, base_targets)
          base_loss.backward()
          # if data_step == 0 and inner_step == 0:
          #     print(f"BEFORE: {network.arch_parameters()}")
          w_optimizer.step()
          # if data_step == 0 and inner_step == 0:
          #     print(f"AFTER: {network.arch_parameters()}")
          w_optimizer.zero_grad()
          
          if args.higher_method in ["val_multiple", "val"]:
              # if data_step < 2 and epoch < 1:
              #   print(f"Arch grads during unrolling from last step: {arch_grads}")
            logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            arch_loss.backward()
            with torch.no_grad():

              for g1, g2 in zip(arch_grads, network.arch_parameters()):
                g1.add_(g2)
              
            network.zero_grad()
            a_optimizer.zero_grad()
            w_optimizer.zero_grad()
            # if data_step < 2 and epoch < 1:
            #   print(f"Arch grads during unrolling: {arch_grads}")
                
      if args.higher_method in ["val_multiple", "val"]:
        print(f"Arch grads after unrolling: {arch_grads}")
        with torch.no_grad():
          for g, p in zip(arch_grads, network.arch_parameters()):
            p.grad = g
                
      a_optimizer.step()
      a_optimizer.zero_grad()
      
      w_optimizer.zero_grad()
      architect.optimizer.zero_grad()
      
      # Restore original model state before unrolling and put in the new arch parameters
      new_arch = deepcopy(network._arch_parameters)
      network.load_state_dict(model_init)
      for p1, p2 in zip(network._arch_parameters, new_arch):
          p1.data = p2.data
          
        
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
          if args.higher_method == "sotl_v2":
            base_inputs, base_targets = arch_inputs, arch_targets ## Use train set for the unrolling to compute hypergradients, then forget the training and train weights only using a separate set
          if data_step in [0, 1] and inner_step < 3 and epoch % 5 == 0:
              logger.info(f"Doing weight training for real in at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
          if args.bilevel_train_steps is not None and inner_step >= args.bilevel_train_steps :
            break
          if args.perturb_alpha:
            # print('before softmax', model.arch_parameters())
            network.softmax_arch_parameters()
                
            # perturb on alpha
            # print('after softmax', model.arch_parameters())
            perturb_alpha(network, base_inputs, base_targets, epsilon_alpha)
            w_optimizer.zero_grad()
            architect.optimizer.zero_grad()
            # print('afetr perturb', model.arch_parameters())
          logits = network(base_inputs)
          base_loss = criterion(logits, base_targets)
          network.zero_grad()
          base_loss.backward()
          w_optimizer.step()
          w_optimizer.zero_grad()
          
          if args.perturb_alpha:
            network.restore_arch_parameters()
          # print('after restore', model.arch_parameters())
          
          n = base_inputs.size(0)

          prec1, prec5 = utils.accuracy(logits, base_targets, topk=(1, 5))

          objs.update(base_loss.item(), n)
          top1.update(prec1.data, n)
          top5.update(prec5.data, n)

      if data_step % args.report_freq == 0:
          logging.info('train %03d %e %f %f', data_step, objs.avg, top1.avg, top5.avg)
      if 'debug' in args.save:
          break

    return  top1.avg, objs.avg


def train(train_queue, valid_queue, network, architect, criterion, w_optimizer, a_optimizer, logger=None, inner_steps=100, epoch=0, 
          steps_per_epoch=None, perturb_alpha=None, epsilon_alpha=None):
    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    train_iter = iter(train_queue)
    valid_iter = iter(valid_queue)
    search_loader_iter = zip(train_iter, valid_iter)
    for data_step, ((base_inputs, base_targets), (arch_inputs, arch_targets)) in tqdm(enumerate(search_loader_iter), total = round(len(train_queue)/inner_steps)):
      if steps_per_epoch is not None and data_step > steps_per_epoch:
        break
      network.train()
      n = base_inputs.size(0)

      base_inputs = base_inputs.cuda()
      base_targets = base_targets.cuda(non_blocking=True)

      # get a random minibatch from the search queue with replacement
      input_search, target_search = next(iter(valid_queue))
      input_search = input_search.cuda()
      target_search = target_search.cuda(non_blocking=True)
      
      all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, 
                                                                                                search_loader_iter, inner_steps=inner_steps, args=args)

      network.zero_grad()

      model_init = deepcopy(network.state_dict())
      w_optim_init = deepcopy(w_optimizer.state_dict())
      arch_grads = [torch.zeros_like(p) for p in network.arch_parameters()]
      for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in tqdm(enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)), desc = "Unrolling bilevel loop", total=inner_steps, disable=True):
          if data_step < 2 and inner_step < 2:
              print(f"Base targets in the inner loop at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
            
          logits = network(base_inputs)
          base_loss = criterion(logits, base_targets)
          base_loss.backward()

          w_optimizer.step()
          a_optimizer.step()

          w_optimizer.zero_grad()
          a_optimizer.zero_grad()
          
                
      a_optimizer.zero_grad()
      
      w_optimizer.zero_grad()
      architect.optimizer.zero_grad()
      
      # Restore original model state before unrolling and put in the new arch parameters
      
    #   new_arch = deepcopy(network._arch_parameters)
    #   network.load_state_dict(model_init)
    #   for p1, p2 in zip(network._arch_parameters, new_arch):
    #       p1.data = p2.data
          
        
    #   for inner_step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(zip(all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets)):
    #       if args.higher_method == "sotl_v2":
    #         base_inputs, base_targets = arch_inputs, arch_targets ## Use train set for the unrolling to compute hypergradients, then forget the training and train weights only using a separate set
    #       if data_step in [0, 1] and inner_step < 3 and epoch % 5 == 0:
    #           logger.info(f"Doing weight training for real in at inner_step={inner_step}, step={data_step}: {base_targets[0:10]}")
    #       if args.bilevel_train_steps is not None and inner_step >= args.bilevel_train_steps :
    #         break
    #       if args.perturb_alpha:
    #         # print('before softmax', model.arch_parameters())
    #         network.softmax_arch_parameters()
                
    #         # perturb on alpha
    #         # print('after softmax', model.arch_parameters())
    #         perturb_alpha(network, base_inputs, base_targets, epsilon_alpha)
    #         w_optimizer.zero_grad()
    #         architect.optimizer.zero_grad()
    #         # print('afetr perturb', model.arch_parameters())
    #       logits = network(base_inputs)
    #       base_loss = criterion(logits, base_targets)
    #       network.zero_grad()
    #       base_loss.backward()
    #       w_optimizer.step()
    #       w_optimizer.zero_grad()
          
    #       if args.perturb_alpha:
    #         network.restore_arch_parameters()
    #       # print('after restore', model.arch_parameters())
          
    #       n = base_inputs.size(0)

    #       prec1, prec5 = utils.accuracy(logits, base_targets, topk=(1, 5))

    #       objs.update(base_loss.item(), n)
    #       top1.update(prec1.data, n)
    #       top5.update(prec5.data, n)

    #   if data_step % args.report_freq == 0:
    #       logging.info('train %03d %e %f %f', data_step, objs.avg, top1.avg, top5.avg)
    #   if 'debug' in args.save:
    #       break

    return  top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            if step > 101:
                break
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                if args.debug:
                    break

    return top1.avg, objs.avg

def format_input_data(base_inputs, base_targets, arch_inputs, arch_targets, search_loader_iter, inner_steps, args, loader_type="train-val"):
    base_inputs, base_targets = base_inputs.cuda(non_blocking=True), base_targets.cuda(non_blocking=True)
    arch_inputs, arch_targets = arch_inputs.cuda(non_blocking=True), arch_targets.cuda(non_blocking=True)
    if args.higher_method == "sotl":
        arch_inputs, arch_targets = None, None
    all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets = [base_inputs], [base_targets], [arch_inputs], [arch_targets]
    for extra_step in range(inner_steps-1):
        if args.inner_steps_same_batch:
            all_base_inputs.append(base_inputs)
            all_base_targets.append(base_targets)
            all_arch_inputs.append(arch_inputs)
            all_arch_targets.append(arch_targets)
            continue # If using the same batch, we should not try to query the search_loader_iter for more samples
        try:
            if loader_type == "train-val" or loader_type == "train-train":
              (extra_base_inputs, extra_base_targets), (extra_arch_inputs, extra_arch_targets)= next(search_loader_iter)
            else:
              extra_base_inputs, extra_base_targets = next(search_loader_iter)
              extra_arch_inputs, extra_arch_targets = None, None
        except Exception as e:
            continue

        extra_base_inputs, extra_base_targets = extra_base_inputs.cuda(non_blocking=True), extra_base_targets.cuda(non_blocking=True)
        if extra_arch_inputs is not None and extra_arch_targets is not None:
          extra_arch_inputs, extra_arch_targets = extra_arch_inputs.cuda(non_blocking=True), extra_arch_targets.cuda(non_blocking=True)
        
        all_base_inputs.append(extra_base_inputs)
        all_base_targets.append(extra_base_targets)
        all_arch_inputs.append(extra_arch_inputs)
        all_arch_targets.append(extra_arch_targets)

    return all_base_inputs, all_base_targets, all_arch_inputs, all_arch_targets

if __name__ == '__main__':
    main()