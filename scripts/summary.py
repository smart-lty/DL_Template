import pandas as pd
import re
import itertools
import ipdb
import os

search_config = {
    "out_channel": [32, 64, 128],
    "num_layers": [2, 3, 4],
    "neg_sample_num":[500, 1000, 2000, 4000, 8000],
    "l2": [1e-2, 5e-3],
    "margin": [0, 1, 1.5, 3]
}

all_parameters = list(itertools.product(*search_config.values()))

res = pd.DataFrame({
    "out_channel":      [],
    "num_layers":       [],
    "neg_sample_num":   [],
    "l2":               [],
    "margin":           [],
    "TEST rank IC":     [],
})


for out_channel, num_layers, neg_sample_num, l2, margin in all_parameters:
    exp_name = f"O_{out_channel}_L_{num_layers}_N_{neg_sample_num}_W_{l2}_M_{margin}"
    try:
        with open(os.path.join("experiments", exp_name, "train.log")) as f:
            content = f.readlines()
            if not content[-1].startswith("Test"):
                continue
            test_rank_ic = float(re.findall(r"0\.[0-9]*", content[-1])[0])
            res.loc[len(res)] = [out_channel, num_layers, neg_sample_num, l2, margin, test_rank_ic]
    except:
        continue

ipdb.set_trace()