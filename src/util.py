import argparse
import torch
import logging
import os
import json
import random
import tqdm
import numpy as np
import pandas as pd


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
    # switches, control some operation
    parser.add_argument("--action1", action="store_true", help="Whether to do action1? ")

    # data related
    parser.add_argument("--data_path", type=str, help="The path where the model will read data.")
    parser.add_argument("--data_param", type=int, help="Feel free to add some parameters related to data.")
    parser.add_argument("--exp_name", "-e", type=str, default="default", help="Where to store the experimental results? ")

    # model related
    parser.add_argument("--model_param", type=int, help="Feel free to add some parameters related to model.")
    
    # train related
    parser.add_argument("--train_param", type=int, help="Feel free to add some parameters related to training.")
    parser.add_argument("--seed", type=int, default=822)
    
    parser.add_argument("--epoch", type=int, default=20)

    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--loss", type=str, default="default")
    parser.add_argument("--metric", type=str, default="default")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=5e-3)

    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2000)
    
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--valid_every", type=int, default=1)
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
    file_handler = logging.FileHandler(os.path.join(exp_name, "train.log"))
    
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(f'{k}: {str(v)}' for k, v in dict(vars(args)).items()))
    logger.info('============================================')
    
    # save config parameters
    with open(os.path.join(exp_name, "params.json"), 'w') as fout:
        json.dump(vars(args), fout)

    
    