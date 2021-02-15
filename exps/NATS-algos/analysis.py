import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
fpath = r"C:\Users\kawga\Documents\Oxford\thesis\AutoDL-Projects\output\search-tss\cifar10\random-affine0_BN0-None\checkpoint\seed-1-corr-metrics.pth"
fpath2=r"output\search-tss\cifar10\random-affine0_BN0-None\seed-1-last-info.pth"
x = torch.load(fpath)

for k,v in x['metrics']["val_accs"].items():
    pass

plt.plot(range(len(v[0])), v[0])

df = pd.DataFrame(v[0])

df.rolling(30, axis=0).mean().plot()