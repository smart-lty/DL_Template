from data import StockPrice
from util import parse_args, initial_experiment, set_rand_seed
from model.mlp import MLP
from model.quantile_mlp import QuantileMLP
from manager.trainer import Trainer
import ipdb
import torch
import os
import json
import tqdm
import logging
import random
import argparse
from torch.utils.data import DataLoader


if __name__ == "__main__":
    
    args = parse_args()
    set_rand_seed(args.seed)
    initial_experiment(args)

    args.device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")

    idxs = list(range(838))

    train_data = StockPrice(index=idxs, mode="train")
    test_data = StockPrice(index=slice(838, None), mode="test")

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_data.collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=test_data.collate_fn)
    
    if args.loss == "quantile":
        model = QuantileMLP(args)
    else:
        model = MLP(args)
    
    if args.init:
        model = torch.load(args.init, map_location=args.device)
        logging.info(f"Loading existing model from {args.init}")
    
    trainer = Trainer(args, model, train_dataloader, test_dataloader)
    
    if args.do_pre:
        trainer.predict(test_dataloader)
    else:
        trainer.train()
    
    