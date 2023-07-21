import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

class Evaluator():
    def __init__(self, top_ratio=1/3, quantile_weight=[5, 9, 8, 7, 6, 5, 4, 3, 2, 1]):
        self.top_ratio = top_ratio
        self.quantile_weight = quantile_weight
        
        self.metrics = ["IC", "MRR", "MR", "HITS"]

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
                
                for metric in self.metrics:
                    eval_res[metric].append(self.compute_metric(score, batch_label, metric))
                
                pbar.update(1)

        eval_res = {k: np.mean(v) for k,v in eval_res.items()}

        return eval_res
    
    def compute_metric(self, pred, ground_truth, metric):
        pred_rank = torch.argsort(torch.argsort(pred, descending=True)) + 1
        gt_rank = torch.argsort(torch.argsort(ground_truth, descending=True)) + 1
        ic = pd.Series(pred_rank.cpu().detach().numpy()).corr(pd.Series(gt_rank.cpu().detach().numpy()))
        
        top_size = int(pred.shape[0] * self.top_ratio)
        pred_at_top_idx = torch.argsort(pred, descending=True)[:top_size]
        pred_at_top_rank = gt_rank[pred_at_top_idx]

        mrr = (1 / pred_at_top_rank).mean().item()
        mr = pred_at_top_rank.float().mean().item()

        precision_at_top = (pred_at_top_rank <= top_size).float().mean().item()

        if metric == "IC":
            return ic
        elif metric == "MRR":
            return mrr
        elif metric == "MR":
            return mr
        elif metric == "HITS":
            return precision_at_top
