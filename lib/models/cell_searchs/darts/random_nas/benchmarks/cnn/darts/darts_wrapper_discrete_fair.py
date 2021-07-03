import sys
# sys.path.append('/nfs/home/robinru/randomNAS_release/darts/cnn')
# sys.path.append('/jmain01/home/JAD029/jph13/rxr49-jph13/randomNAS_release/darts/cnn')
# sys.path.append(r'C:\Users\miros\Documents\Oxford\AutoDL-Projects\lib\models\cell_searchs\darts\data')
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / '..' / '..' / 'cnn').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
import genotypes
from model_search import Network_orig
import utils

import time
import math
import copy
import random
import logging
import os
import gc
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from operator import add

from genotypes import PRIMITIVES

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper:
    def __init__(self, save_path, seed, batch_size, grad_clip, epochs, resume_iter=None, init_channels=16):
        args = {}
        # args['data'] = '/jmain01/home/JAD029/jph13/rxr49-jph13/randomNAS_release/darts/data/'
        args['data'] = r'C:\Users\miros\Documents\Oxford\AutoDL-Projects\lib\models\cell_searchs\darts\data'
        # args['data'] = '/nfs/home/robinru/randomNAS_release/darts/data/'

        args['epochs'] = epochs
        args['learning_rate'] = 0.025
        args['batch_size'] = batch_size
        args['learning_rate_min'] = 0.001
        args['momentum'] = 0.9
        args['weight_decay'] = 3e-4
        args['init_channels'] = init_channels
        args['layers'] = 8
        args['drop_path_prob'] = 0.3
        args['grad_clip'] = grad_clip
        args['train_portion'] = 0.5
        args['seed'] = seed
        args['log_interval'] = 50
        args['save'] = save_path
        args['gpu'] = 0
        args['cuda'] = True
        args['cutout'] = False
        args['cutout_length'] = 16
        args['report_freq'] = 50
        args = AttrDict(args)
        self.args = args
        self.seed = seed

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = False
        cudnn.enabled=True
        cudnn.deterministic=True
        torch.cuda.manual_seed_all(args.seed)


        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        self.train_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
          pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        self.valid_queue = torch.utils.data.DataLoader(
          train_data, batch_size=args.batch_size,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
          pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        self.train_iter = iter(self.train_queue)
        self.valid_iter = iter(self.valid_queue)

        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        self.criterion = criterion

        model = Network_orig(args.init_channels, 10, args.layers, self.criterion)

        model = model.cuda()
        self.model = model

        try:
            self.load()
            logging.info('loaded previously saved weights')
        except Exception as e:
            print(e)

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
          self.model.parameters(),
          args.learning_rate,
          momentum=args.momentum,
          weight_decay=args.weight_decay)
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        if resume_iter is not None:
            self.steps = resume_iter
            self.epochs = int(resume_iter / len(self.train_queue))
            logging.info("Resuming from epoch %d" % self.epochs)
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
            for i in range(self.epochs):
                self.scheduler.step()

        size = 0
        for p in model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))

        total_params = sum(x.data.nelement() for x in model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

    def train_batch(self, arch_list):
      args = self.args
      if self.steps % len(self.train_queue) == 0:
        self.scheduler.step()
        self.objs = utils.AvgrageMeter()
        self.top1 = utils.AvgrageMeter()
        self.top5 = utils.AvgrageMeter()
      lr = self.scheduler.get_lr()[0]
      step = self.steps % len(self.train_queue)
      input, target = next(self.train_iter)
      n = input.size(0)
      input = Variable(input, requires_grad=False).cuda()
      target = Variable(target, requires_grad=False).cuda()

      for j, arch in enumerate(arch_list):
        weights = self.get_weights_from_arch(arch)
        self.set_model_weights(weights)
        self.model.train()

        # get a random minibatch from the search queue with replacement
        self.optimizer.zero_grad()
        logits = self.model(input, discrete=True)
        loss = self.criterion(logits, target)/len(arch_list)
        loss.backward()

      nn.utils.clip_grad_norm(self.model.parameters(), args.grad_clip)
      self.optimizer.step()

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      self.objs.update(loss.data, n)
      self.top1.update(prec1.data, n)
      self.top5.update(prec5.data, n)

      if step % args.report_freq == 0:
        logging.info('train %03d %e %f %f', step, self.objs.avg, self.top1.avg, self.top5.avg)

      self.steps += 1
      if self.steps % len(self.train_queue) == 0:
        self.epochs += 1
        self.train_iter = iter(self.train_queue)
        valid_err = self.evaluate(arch)
        logging.info('epoch %d  |  train_acc %f  |  valid_acc %f' % (self.epochs, self.top1.avg, 1-valid_err))
        self.save()

    def evaluate(self, arch, split=None, eval_metric='val_acc'):

        if eval_metric == 'val_acc':
            # Return error since we want to minimize obj val
            logging.info(arch)
            objs = utils.AvgrageMeter()
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()

            weights = self.get_weights_from_arch(arch)
            self.set_model_weights(weights)

            self.model.eval()

            if split is None:
                n_batches = 100
            else:
                n_batches = len(self.valid_queue)

            self.valid_iter = iter(self.valid_queue)
            with torch.no_grad():
                for step in range(n_batches):
                    try:
                        input, target = next(self.valid_iter)
                    except Exception as e:
                        logging.info('looping back over valid set')
                        self.valid_iter = iter(self.valid_queue)
                        input, target = next(self.valid_iter)

                    input = Variable(input).cuda()
                    target = Variable(target).cuda()

                    logits = self.model(input, discrete=True)
                    loss = self.criterion(logits, target)

                    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                    n = input.size(0)
                    objs.update(loss.data, n)
                    top1.update(prec1.data, n)
                    top5.update(prec5.data, n)

                    if step % self.args.report_freq == 0:
                        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

            return 1 - top1.avg

        elif eval_metric == 'sotltrain':
            args = self.args

            # Return error since we want to minimize obj val
            logging.info(arch)
            objs = utils.AvgrageMeter()
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()

            weights = self.get_weights_from_arch(arch)
            self.set_model_weights(weights)
            model_sample = copy.deepcopy(self.model)
            optimizer_eval = torch.optim.SGD(
                model_sample.parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)

            scheduler_eval = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_eval, float(args.epochs), eta_min=args.learning_rate_min)

            # TODO I obtain this value by running the following 2 lines of code above but I cannot remember the exact
            # iterations I ask the scheduler to step so I checked my log and find the particular lr value is 1.3e-2
            for _ in range(50):
                scheduler_eval.step()

            curr_lr = scheduler_eval.get_lr()[0]
            model_sample.train()

            if split is None:
                n_batches = 102
            else:
                n_batches = len(self.train_queue)

            train_losses_list = []
            valid_acc_list = []
            valid_loss_list = []
            valid_acc_mini_list = []
            valid_loss_mini_list = []
            
            for step in range(n_batches):

                try:
                    input, target = next(self.train_iter)
                except Exception as e:
                    logging.info('looping back over valid set')
                    self.train_iter = iter(self.train_queue)
                    input, target = next(self.train_iter)

                input = Variable(input).cuda()
                target = Variable(target).cuda()

                optimizer_eval.zero_grad()
                logits = model_sample(input, discrete=True)
                loss = self.criterion(logits, target)
                loss.backward()

                nn.utils.clip_grad_norm(model_sample.parameters(), args.grad_clip)
                optimizer_eval.step()

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
                train_losses_list.append(loss.data.cpu().numpy())
                val_loss_whole, val_acc_whole, val_acc_list, val_loss_list  = self.validate(model_sample, 1)

                if step % 100 == 0:
                    val_loss_whole, val_acc_whole, val_acc_list, val_loss_list  = self.validate(model_sample, 100) # Use 100 minibatches as estimate of total SoVL
                    model_sample.train()
                    valid_loss_list.append(float(val_loss_whole.cpu().numpy()))
                    valid_acc_list.append(float(val_acc_whole.cpu().numpy()))
                    valid_loss_mini_list.append(float(val_loss_list[0].cpu().numpy())) 
                    valid_acc_mini_list.append(float(val_acc_list[0].cpu().numpy()))
                else:
                    valid_acc_mini_list.append(valid_acc_mini_list[-1])
                    valid_loss_mini_list.append(valid_loss_mini_list[-1])
                    valid_loss_list.append(valid_loss_list[-1])
                    valid_acc_list.append(valid_acc_list[-1])
                    
                if step % self.args.report_freq == 0:
                    logging.info('sotl %03d %e %e', step, objs.avg, curr_lr)

            return train_losses_list, valid_loss_list, valid_acc_list, valid_loss_mini_list, valid_acc_mini_list

    def validate(self, model, n_batches):
        # evaluate on valid set
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        model.eval()
        valid_iter = iter(self.valid_queue)
        val_accs, val_losses = [], []
        with torch.no_grad():
            for step in range(n_batches):
                try:
                    input, target = next(valid_iter)
                except Exception as e:
                    logging.info('looping back over valid set')
                    self.valid_iter = iter(self.valid_queue)
                    input, target = next(valid_iter)

                input = Variable(input).cuda()
                target = Variable(target).cuda()

                logits = model(input, discrete=True)
                loss = self.criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
                
                val_accs.append(prec1)
                val_losses.append(loss)
        model.train()
        return objs.avg, top1.avg, val_accs, val_losses

    def save(self):
        utils.save(self.model, os.path.join(self.args.save, 'weights.pt'))

    def load(self):
        utils.load(self.model, os.path.join(self.args.save, 'weights.pt'))

    def get_weights_from_arch(self, arch):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)
        alphas_reduce = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=False)

        offset = 0
        for i in range(n_nodes):
            normal1 = arch[0][2*i]
            normal2 = arch[0][2*i+1]
            reduce1 = arch[1][2*i]
            reduce2 = arch[1][2*i+1]
            alphas_normal[offset+normal1[0], normal1[1]] = 1
            alphas_normal[offset+normal2[0], normal2[1]] = 1
            alphas_reduce[offset+reduce1[0], reduce1[1]] = 1
            alphas_reduce[offset+reduce2[0], reduce2[1]] = 1
            offset += (i+2)

        arch_parameters = [
          alphas_normal,
          alphas_reduce,
        ]
        return arch_parameters

    def set_model_weights(self, weights):
      self.model.alphas_normal = weights[0]
      self.model.alphas_reduce = weights[1]
      self.model._arch_parameters = [self.model.alphas_normal, self.model.alphas_reduce]

    def sample_arch(self):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps

        normal = []
        reduction = []
        
        normal_nb = []
        reduction_nb = []
        for i in range(n_nodes):
            nonzero_ops = [i for i in range(num_ops) if PRIMITIVES[i] != "none"]
            
            ops = np.random.choice(nonzero_ops, 4)
            nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
            nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
            normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
            reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])

            normal_nb.extend([(PRIMITIVES[ops[0]], nodes_in_normal[0]), (PRIMITIVES[ops[1]], nodes_in_normal[1])])
            reduction_nb.extend([(PRIMITIVES[ops[2]], nodes_in_reduce[0]), (PRIMITIVES[ops[3]], nodes_in_reduce[1])])

        return (normal, reduction), (normal_nb, reduction_nb)

    def sample_arch_fair(self):
        k = sum(1 for i in range(self.model._steps) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = self.model._steps
        arch_list = []

        # Fair Sampling
        rngs = []
        seed = self.seed
        for j in range(4*n_nodes):
            seed += 100
            random.seed(seed)
            rngs.append(random.sample(range(num_ops), num_ops))

        rngs = np.vstack(rngs).T
        for j in range(num_ops):
            normal = []
            reduction = []
            for i in range(n_nodes):
                # ops = np.random.choice(range(num_ops), 4)
                ops = rngs[j, i*4:i*4+4]
                nodes_in_normal = np.random.choice(range(i+2), 2, replace=False)
                nodes_in_reduce = np.random.choice(range(i+2), 2, replace=False)
                normal.extend([(nodes_in_normal[0], ops[0]), (nodes_in_normal[1], ops[1])])
                reduction.extend([(nodes_in_reduce[0], ops[2]), (nodes_in_reduce[1], ops[3])])
            arch_j = (normal, reduction)
            arch_list.append(arch_j)

        return arch_list

    def perturb_arch(self, arch):
        new_arch = copy.deepcopy(arch)
        num_ops = len(genotypes.PRIMITIVES)

        cell_ind = np.random.choice(2)
        step_ind = np.random.choice(self.model._steps)
        nodes_in = np.random.choice(step_ind+2, 2, replace=False)
        ops = np.random.choice(range(num_ops), 2)

        new_arch[cell_ind][2*step_ind] = (nodes_in[0], ops[0])
        new_arch[cell_ind][2*step_ind+1] = (nodes_in[1], ops[1])
        return new_arch


