import numpy as np
import os
import itertools
import ipdb

search_config = {
    "loss": ["mse", "tr"],
    "reg": [0.1, 0.2, 0.3, 0.4],
    "neg_sample_num":[500, 1000, 2000, 4000],
}


all_parameters = list(itertools.product(*search_config.values()))
pnlfiledir = "./pnl/"
alphaname = "task1_backtest_"
for loss, reg, neg in all_parameters:
    exp_name = f"grid_search/loss_{loss}_reg_{reg}_neg_{neg}"
    
    cmd = f"python src/main.py --gpu 4 --do_pre --init experiments/{exp_name}"
    os.system(cmd)

    idx = 0
    while idx < 1000 :
        if os.path.isdir(pnlfiledir + alphaname + str(idx)):
            idx += 1
        else:
            break
    
    cmd = f"python /home/tianyu/backtest/long100.risk.py /data203/tianyu/task1_backtest"
    os.system(cmd)

    os.rename(os.path.join(pnlfiledir, alphaname + str(idx)),os.path.join(pnlfiledir, exp_name))
