import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets import get_datasets, SearchDataset
from procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from utils import get_model_infos, obtain_accuracy
from log_utils import AverageMeter, time_string, convert_secs2time
from models import get_search_spaces
from nats_bench import create
from typing import *
import wandb


def wandb_auth(fname: str = "nas_key.txt"):
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
    wandb.login()


def simulate_train_eval_sotl(
    api,
    arch,
    dataset,
    iepoch: Optional[int] = None,
    hp: str = "12",
    account_time: bool = True,
    metric: str = "valid-accuracy",
    e: int = 1,
    is_random: bool = True,
):
    """This function is used to simulate training and evaluating an arch."""
    index = api.query_index_by_arch(arch)
    all_names = ("cifar10", "cifar100", "ImageNet16-120")
    if dataset not in all_names:
        raise ValueError("Invalid dataset name : {:} vs {:}".format(dataset, all_names))
    if dataset == "cifar10":
        dataset = "cifar10-valid"

    if e > 1 and "loss" in metric:
        losses = []
        for i in range(iepoch - e + 1, iepoch + 1):
            info = api.get_more_info(
                index, dataset, iepoch=i, hp=hp, is_random=is_random
            )
            losses.append(info[metric])

        observed_metric, time_cost = (
            sum(losses),
            info["train-all-time"] + info["valid-per-time"],
        )

    else:
        info = api.get_more_info(
            index, dataset, iepoch=iepoch, hp=hp, is_random=is_random
        )
        observed_metric, time_cost = (
            info[metric],
            info["train-all-time"] + info["valid-per-time"],
        )
    if metric == "train-loss":
        observed_metric = -observed_metric
    latency = api.get_latency(index, dataset)
    if account_time:
        api._used_time += time_cost
    return observed_metric, latency, time_cost, api._used_time


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
