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

gpus = [1, 3]
gpu_count = 0

f = open("scripts/train.sh", "w")


for loss, reg, neg in all_parameters:
    exp_name = f"grid_search/loss_{loss}_reg_{reg}_neg_{neg}"
    
    cmd = f"nohup python src/main.py --gpu {gpus[gpu_count]} -e {exp_name} --loss {loss} --reg {reg}  --neg_sample_num {neg} > /dev/null"
    if (gpu_count + 1) % (3 * len(gpus)) == 0:
        cmd += "2>&1 \n\n"
    else:
        cmd += " 2>&1 &\n"

    f.write(cmd)
    gpu_count = (gpu_count + 1) % len(gpus)

