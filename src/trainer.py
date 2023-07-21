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
from loss import Loss
from regularizer import Regularizer
from util import plot_density


class Trainer():
    def __init__(self, args, model, train_dataloader, valid_dataloader=None, test_dataloader=None):
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.device = self.args.device
        self.updates_counter = 0

        model_params = list(self.model.parameters())
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if args.optimizer == "SGD":
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_params), lr=args.lr, weight_decay=self.args.l2)
        if args.optimizer == "Adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_params), lr=args.lr, weight_decay=self.args.l2)
        if args.optimizer == "AdamW":
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_params), lr=args.lr, weight_decay=self.args.l2)

        self.criterion = Loss(args)
        self.regularizer = Regularizer(args)
        
        self.model = self.model.to(self.device)
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = -100
        self.stop_training = False

    def train_epoch(self):
        total_loss = []
        all_labels = []
        all_scores = []

        dataloader = self.train_dataloader
        self.model.train()

        tic = time.time()
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                batch_data, batch_label = batch
                
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)

                self.optimizer.zero_grad()
                score, features = self.model(batch_data)
                score = score.squeeze(1)

                loss1 = self.criterion(score, batch_label, sample_num=self.args.neg_sample_num)
                loss2 = self.regularizer(features, batch_label, self.args.neg_sample_num)

                loss = (1 - self.args.reg) * loss1 + self.args.reg * loss2

                loss.backward()

                total_loss.append(loss.item())

                self.optimizer.step()
                pbar.update(1)

                if self.args.do_plot and pbar.n % 100 == 0:
                    plot_density(score, batch_label, self.args.exp_name, self.epoch, pbar.n)

        logging.info(f'Epoch {self.epoch} Loss:{np.mean(total_loss)} in {str(time.time() - tic)} s ')

        return np.mean(total_loss)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        rank_IC = []
        rank_MRR = []
        rank_MR = []
        rank_HITS = []

        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                batch_data, batch_label = batch
                
                batch_data = batch_data.to(self.device)
                batch_label = batch_label.to(self.device)

                score, _ = self.model(batch_data)
                score = score.squeeze(1)
                
                rank_IC.append(self.compute_IC(score, batch_label))
                rank_MRR.append(self.compute_MRR(score, batch_label))
                rank_MR.append(self.compute_MR(score, batch_label))
                rank_HITS.append(self.compute_Hits(score, batch_label))
                pbar.update(1)

        metrics = {"rank_IC": np.mean(rank_IC), "rank_MRR": np.mean(rank_MRR), "rank_MR": 1 / np.mean(rank_MR), "rank_HITS": np.mean(rank_HITS)}
        return metrics

    def compute_IC(self, pred, ground_truth):
        pred_rank = pd.Series(pred.cpu().detach().numpy()).rank(ascending=False)
        gt_rank = pd.Series(ground_truth.cpu().detach().numpy()).rank(ascending=False)
        ic = pred_rank.corr(gt_rank)
        return ic
    
    def compute_MRR(self, pred, ground_truth):
        top_size = pred.shape[0] // 3
        idx = torch.argsort(pred, descending=True)[:top_size]
        gt_rank = torch.argsort(torch.argsort(ground_truth, descending=True)) + 1
        pred_rank = gt_rank[idx]
        mrr = (1 / pred_rank).mean().item()
        return mrr

    def compute_MR(self, pred, ground_truth):
        top_size = pred.shape[0] // 3
        idx = torch.argsort(pred, descending=True)[:top_size]
        gt_rank = torch.argsort(torch.argsort(ground_truth, descending=True)) + 1
        pred_rank = gt_rank[idx].float()
        mr = pred_rank.mean().item()
        return mr
    
    def compute_Hits(self, pred, ground_truth):
        top_size = pred.shape[0] // 3
        idx = torch.argsort(pred, descending=True)[:top_size]
        gt_rank = torch.argsort(torch.argsort(ground_truth, descending=True)) + 1
        pred_rank = gt_rank[idx].float()
        precision = (pred_rank <= top_size).float().mean().item()

        idx = torch.argsort(ground_truth)[:top_size]
        pred_rank = torch.argsort(torch.argsort(ground_truth)) + 1
        gt_rank = pred_rank[idx].float()
        recall = (gt_rank <= top_size).float().mean().item()

        # return 2 * recall * precision / (recall + precision)
        return precision

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.args.epoch + 1):
            self.epoch = epoch
            self.train_epoch()
            
            if self.valid_dataloader and epoch % self.args.valid_every == 0:
                tic = time.time()
                metrics = self.evaluate(self.valid_dataloader)
                logging.info(f'Epoch {self.epoch} Valid Performance:{metrics} in {str(time.time() - tic)} s ')
                res = metrics["rank_"+self.args.metric]
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
        logging.info(f'Epoch {self.epoch} Better models found w.r.t {self.args.metric}. Saved it!')

    @torch.no_grad()
    def predict(self, test_data):
        """Predict the next two-day price difference of all stocks for a given day."""
        self.model.eval()
        # generate test results for backtest
        for i in tqdm.trange(len(test_data), desc="generating backtest"):
            day = test_data.data[i]
            label = test_data.label[i]
            # get all stocks id from file "codelist"
            all_stocks = pd.read_csv(os.path.join(test_data.train_path, day, "codelist"), header=None).iloc[:, 0]
            # only select stocks which appear in both training and inference
            all_stocks = pd.merge(all_stocks, label)

            all_stocks_name = all_stocks[0].apply(lambda x: str(x).rjust(6, "0")+".npy")
            all_stocks_data = torch.from_numpy(np.stack([np.load(os.path.join(test_data.train_path, day, stock)) for stock in all_stocks_name])).to(self.device)
            stocks_score, _ = self.model(all_stocks_data)
            stocks_score = stocks_score.squeeze(1).detach().cpu().numpy()
            all_stocks[1] = pd.Series(stocks_score)
            all_stocks.to_csv(f"/data203/tianyu/task1_backtest/{day}", header=False, index=False)