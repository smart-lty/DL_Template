import argparse
import torch
import logging
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_rand_seed(seed=1):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def parse_args():
    parser = argparse.ArgumentParser()
    # data related
    parser.add_argument("--do_pre", action="store_true")
    parser.add_argument("--do_plot", action="store_true")

    # model related
    parser.add_argument("--in_channel", type=int, default=590)
    parser.add_argument("--out_channel", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--pe", type=str, default="bert")
    
    # train related
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--metric", type=str, default="HITS")
    parser.add_argument("--neg_sample_num", type=int, default=2000)
    parser.add_argument("--exp_name", "-e", type=str, default="test")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--reg", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=5e-3)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--valid_every", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--init", type=str, default="") 
    parser.add_argument("--seed", type=int, default=822)
    args = parser.parse_args()
    return args

def initial_experiment(args):
    """Initialize an experiment"""
    logging.basicConfig(level=logging.INFO)
    
    exp_path = "./experiments"
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    exp_name = os.path.join(exp_path, args.exp_name)
    if not os.path.exists(exp_name):
        os.mkdir(exp_name)
    args.exp_name = exp_name
    if args.do_pre:
        file_handler = logging.FileHandler(os.path.join(exp_name, "predict.log"))
    else:
        file_handler = logging.FileHandler(os.path.join(exp_name, "train.log"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(f'{k}: {str(v)}' for k, v in dict(vars(args)).items()))
    logger.info('============================================')
    with open(os.path.join(exp_name, "params.json"), 'w') as fout:
        json.dump(vars(args), fout)

def plot_density(pred, ground_truth, exp_name, epoch, idx):
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(1)
    if len(ground_truth.shape) == 1:
        ground_truth = ground_truth.unsqueeze(1)
    
    score_diff = pred - pred.T
    gt_diff = ground_truth - ground_truth.T
    
    plt.figure(figsize=(10, 10))
    df = pd.DataFrame({
        "pred": score_diff.view(-1).cpu().detach().numpy(),
        "gt": gt_diff.view(-1).cpu().detach().numpy()
    })
    df.plot.kde()
    plt.savefig(f"figures/{exp_name}_{epoch}_{idx}.jpg")
    
    