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
from manager.evaluator import Evaluator


class Trainer():
    def __init__(self, args, model, train_dataloader, valid_dataloader=None, test_dataloader=None):
        self.args = args
        
        self.device = self.args.device
        
        self.model = model.to(self.device)
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        model_params = list(self.model.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if args.optimizer == "SGD":
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_params), lr=args.lr, weight_decay=self.args.l2)
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_params), lr=args.lr, weight_decay=self.args.l2)
        if args.optimizer == "AdamW":
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_params), lr=args.lr, weight_decay=self.args.l2)

        self.criterion = Loss(args)
        self.evaluator = Evaluator(args)
        
        self.updates_counter = 0
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = -100
        self.stop_training = False

    def train_epoch(self):
        total_loss = []

        dataloader = self.train_dataloader
        self.model.train()

        tic = time.time()
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                batch_data, batch_label = batch
                
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)

                self.optimizer.zero_grad()

                # feed forward
                score = self.model(batch_data)

                # compute loss
                loss = self.criterion(score, batch_label)

                # feed backward
                loss.backward()

                total_loss.append(loss.item())

                self.optimizer.step()
                pbar.update(1)

        logging.info(f'Epoch {self.epoch} Loss:{np.mean(total_loss)} in {str(time.time() - tic)} s ')

        return np.mean(total_loss)

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.args.epoch + 1):
            self.epoch = epoch
            self.train_epoch()
            
            if self.valid_dataloader and epoch % self.args.valid_every == 0:
                tic = time.time()
                metrics = self.evaluator.evaluate(self.model, self.valid_dataloader, self.device)

                logging.info(f'Epoch {self.epoch} Valid Performance:{metrics} in {str(time.time() - tic)} s ')
                
                res = metrics[self.args.metric]
                
                if res >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = res
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                    if self.not_improved_count >= self.args.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.args.early_stop} epochs. Training stops.")
                        break
            
            logging.info("=" * 100)
    
    def test(self):
        if not self.test_dataloader:
            logging.info("No test data is given.")
            return

        tic = time.time()
        self.model = torch.load(os.path.join(self.args.exp_name, "best_graph_classifier.pth"))
        self.model = self.model.to(self.device)
        metrics = self.evaluate(self.test_dataloader)
        
        logging.info(f'Test Performance:{metrics} in {str(time.time() - tic)} s ')

    def save_classifier(self):
        torch.save(self.model, os.path.join(self.args.exp_name, 'best_graph_classifier.pth'))  
        logging.info(f'Epoch {self.epoch} Better models found w.r.t IC. Saved it!')