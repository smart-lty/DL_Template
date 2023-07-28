import torch
import os
import tqdm
import logging
import random
from torch.utils.data import DataLoader
from data import UserDataset
from util import parse_args, initial_experiment, set_rand_seed
from model.base_model import BaseModel
from manager.trainer import Trainer


if __name__ == "__main__":
    
    args = parse_args()
    set_rand_seed(args.seed)
    initial_experiment(args)

    args.device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")

    train_data = UserDataset(args)
    valid_data = UserDataset(args)
    test_data = UserDataset(args)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=UserDataset.collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=UserDataset.collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=UserDataset.collate_fn)
    
    model = BaseModel(args)
    
    trainer = Trainer(args, model, train_dataloader, valid_dataloader, test_dataloader)
    
    trainer.train()
    trainer.test()
    
    