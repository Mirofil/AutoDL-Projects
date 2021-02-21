import multiprocessing
import multiprocess as mp
import os
from utils.sotl_utils import wandb_auth
import wandb
from tqdm import tqdm

def reset_wandb_env():
    # https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-cross-validation/train-cross-validation.py good example on launching multiple Runs from Sweep-initiated runs
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

def train_stats_reporter(queue,config,sweep_group,sweep_run_name, arch):
  reset_wandb_env()
  wandb_auth()

  run = wandb.init(
          group=sweep_group,
          project="NAS",
          config={**config, "arch":arch},
      )

  while True:
    elem = queue.get()
    if elem == "SENTINEL":
      break
    wandb.log(elem)

  wandb.join()
  run.finish()



def log_train_stats_per_arch(train_stats, config, sweep_group, sweep_run_name):
    # This mass logging at the end doesnt work and I dont know why
  reset_wandb_env()
  wandb_auth()

  for arch in tqdm(train_stats.keys(), desc="Iterating over train statistics per arch"):

    run = wandb.init(
            group=sweep_group,
            job_type=sweep_run_name,
            config={**config, "arch":arch},
        )
    for batch_train_stats in train_stats[arch]:
      print("ITERATION!")
      print(batch_train_stats)
      wandb.log(batch_train_stats)

    wandb.join()
    run.finish()
  
  pass
