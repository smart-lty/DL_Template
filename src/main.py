from data import StockPrice
from util import parse_args, initial_experiment, set_rand_seed
from model import Model, PNAModel
from trainer import Trainer
import ipdb
import torch
import os
import tqdm
import logging
import random
from torch.utils.data import DataLoader


if __name__ == "__main__":
    
    args = parse_args()
    set_rand_seed(args.seed)
    initial_experiment(args)
    args.device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")

    idxs = list(range(838))
    random.shuffle(idxs)

    train_index = idxs[:638]
    valid_index = idxs[638:]

    train_data = StockPrice(index=train_index, mode="train")
    valid_data = StockPrice(index=valid_index, mode="valid")
    test_data = StockPrice(index=slice(838, None), mode="test")

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_data.collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=valid_data.collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=test_data.collate_fn)
    
    model = Model(args)
    
    if args.init:
        model = torch.load(os.path.join(args.init, "best_graph_classifier.pth"))
        logging.info(f"Loading existing model from {args.init}")
    
    trainer = Trainer(args, model, train_dataloader, valid_dataloader, test_dataloader)
    if args.do_pre:
        trainer.predict(test_data)
    else:
        trainer.train()
        trainer.test()
    
    