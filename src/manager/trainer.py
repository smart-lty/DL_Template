import os
import logging
import time
import ipdb
import tqdm
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loss.loss import Loss
from loss.regularizer import Regularizer
from util import plot_density

from manager.evaluator import Evaluator


class Trainer():
    def __init__(self, args, model, train_dataloader, test_dataloader=None):
        
        self.args = args
        self.device = self.args.device
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        model_params = list(self.model.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if args.optimizer == "SGD":
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_params), lr=args.lr, weight_decay=self.args.l2)
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_params), lr=args.lr, weight_decay=self.args.l2)

        self.criterion = Loss(args)
        self.regularizer = Regularizer(args)

        self.evaluator = Evaluator()

    def train_epoch(self):
        total_loss = []
        all_labels = []
        all_scores = []

        dataloader = self.train_dataloader
        self.model.train()

        tic = time.time()
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                self.optimizer.zero_grad()

                batch_data, batch_label = batch
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)

                score, factors = self.model(batch_data)

                loss = self.criterion(score, batch_label)
                reg = self.regularizer(factors, batch_label, self.args.neg_sample_num)
                loss += reg
                loss.backward()

                total_loss.append(loss.item())

                self.optimizer.step()
                pbar.update(1)

        logging.info(f'Epoch {self.epoch} Loss:{np.mean(total_loss)} in {str(time.time() - tic)} s ')

        return np.mean(total_loss)

    def train(self):
        for epoch in range(1, self.args.epoch + 1):
            self.epoch = epoch
            self.train_epoch()
            
            if self.test_dataloader and epoch % self.args.valid_every == 0:
                tic = time.time()
                metrics = self.evaluator.evaluate(self.model, self.test_dataloader, self.device)
                logging.info(f'Epoch {self.epoch} Test Performance:{metrics} in {str(time.time() - tic)} s ')
                torch.save(self.model, os.path.join(self.args.exp_name, f"epoch_{self.epoch}.pth"))
            
            logging.info("=" * 100)

    @torch.no_grad()
    def predict(self, dataloader):
        """Predict the next two-day price difference of all stocks for a given day."""
        self.model.eval()

        if not os.path.exists(os.path.join(os.path.dirname(self.args.init), "backtest_results")):
            os.mkdir(os.path.join(os.path.dirname(self.args.init), "backtest_results"))

        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for i, batch in enumerate(dataloader):
                batch_data, batch_label, batch_stock = batch
                
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)

                score, _ = self.model(batch_data)
                
                if len(score.shape) != 1:
                    score = score.softmax(dim=1)
                    score = score * torch.tensor(self.quantile_weight, device=score.device)
                    score = score.mean(dim=1)

                df = pd.DataFrame({"stock": batch_stock, "score": pd.Series(score.cpu())})
                df.to_csv(os.path.join(os.path.dirname(self.args.init), "backtest_results", self.test_dataloader.dataset.data[i]), header=False, index=False)

                pbar.update(1)