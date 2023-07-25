import tqdm
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ipdb

class Evaluator():
    def __init__(self, top_ratio=1/3, quantile_weight=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]):
        self.top_ratio = top_ratio
        self.quantile_weight = quantile_weight
        
        self.metrics = ["IC", "MRR", "MR", "HITS", "ER"]
        self.table = torch.tensor(self.preprocess_quantile_table().values)

    def preprocess_quantile_table(self, num_quantile=500, label_path="/data203/tianyu/d2r"):
        pre_path = os.path.dirname(label_path)
        
        if os.path.exists(os.path.join(pre_path, f"quantile_table.csv")):
            table = pd.read_csv(os.path.join(pre_path, f"quantile_table.csv"), header=None)[0]
            return table

        day_list = os.listdir(label_path)
        label_list = [pd.read_csv(os.path.join(label_path, day), header=None)[1] for day in day_list]

        table = 0

        for label in tqdm.tqdm(label_list, total=len(label_list)):
            new_label = label - label.mean()
            new_label = new_label.sort_values(ascending=False).reset_index(drop=True)

            quantile_length = int(new_label.index.shape[0] / num_quantile) + 1
            group = pd.Series(new_label.index // quantile_length)
            new_label = pd.DataFrame({"value": new_label, "group": group})
            new_label = new_label.groupby("group")["value"].apply(lambda x: x.mean())
            
            for i in range(num_quantile - new_label.shape[0]):
                new_label.loc[len(new_label)] = new_label.loc[len(new_label) - 1]
        
            table += new_label
        table /= len(label_list)

        table.to_csv(os.path.join(pre_path, f"quantile_table.csv"), header=None, index=False)
        return table

    @torch.no_grad()
    def evaluate(self, model, dataloader, device):
        model.eval()

        eval_res = {metric: [] for metric in self.metrics}

        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                batch_data, batch_label = batch
                
                batch_data = batch_data.to(device)
                batch_label = batch_label.to(device)

                score, _ = model(batch_data)
                
                if len(score.shape) != 1:
                    score = score.softmax(dim=1)
                    score = score * torch.tensor(self.quantile_weight, device=score.device)
                    score = score.mean(dim=1)

                batch_metric = self.compute_metric(score, batch_label)

                for metric in self.metrics:
                    eval_res[metric].append(batch_metric[metric])
                
                pbar.update(1)

        eval_res = {k: np.mean(v) for k,v in eval_res.items()}

        return eval_res
    
    def compute_metric(self, pred, ground_truth):
        pred_rank = torch.argsort(torch.argsort(pred, descending=True)) + 1
        gt_rank = torch.argsort(torch.argsort(ground_truth, descending=True)) + 1
        # compute IC
        ic = pd.Series(pred_rank.cpu().detach().numpy()).corr(pd.Series(gt_rank.cpu().detach().numpy()))
        
        top_size = int(pred.shape[0] * self.top_ratio)
        pred_at_top_idx = torch.argsort(pred, descending=True)[:top_size]
        pred_at_top_rank = gt_rank[pred_at_top_idx]

        # compute MRR and MR
        mrr = (1 / pred_at_top_rank).mean().item()
        mr = pred_at_top_rank.float().mean().item()

        # compute HITS
        precision_at_top = (pred_at_top_rank <= top_size).float().mean().item()

        # compute Expected Return (ER)
        num_quantile = self.table.shape[0]
        quantile_length = int(pred.shape[0] / num_quantile) + 1
        
        pred_at_top_group = torch.div(pred_at_top_rank, quantile_length, rounding_mode="floor")
        
        ER = self.table[pred_at_top_group.cpu()].mean()

        return {"IC": ic, "MRR": mrr, "MR": mr, "HITS": precision_at_top, "ER": ER}
