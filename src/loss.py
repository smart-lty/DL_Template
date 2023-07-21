import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import util
from audtorch.metrics.functional import pearsonr


class Loss(nn.Module):
    """Implement a unique loss"""
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.top_k = args.top_k
        self.margin = args.margin
        self.margin_loss = nn.MarginRankingLoss(margin=args.margin)
        self.quantiles = [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.75, 0.9, 0.95]

    
    def forward(self, pred, ground_truth, sample_num):
        if self.args.loss == "mse":
            return self.mse_loss(pred, ground_truth)
        elif self.args.loss == "wl":
            return self.wranknet_loss(pred, ground_truth)
        elif self.args.loss == "tr":
            return self.total_ranking_loss(pred, ground_truth)
        elif self.args.loss == "logcos":
            return self.logcos_loss(pred, ground_truth)
        elif self.args.loss == "quantile":
            return self.quantile_loss(pred, ground_truth)
        # return self.sample_ranking_loss(pred, ground_truth, sample_num) + self.mse_loss(pred, ground_truth) + self.corr_pe2_loss(pred, ground_truth)
        # return self.total_ranking_loss(pred, ground_truth, sample_num)
    
    def quantile_loss(self, pred, ground_truth):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = ground_truth - pred[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = torch.cat(losses, dim=2).mean()

        return losses

    def mse_loss(self, pred, ground_truth):
        return F.mse_loss(pred.float(), ground_truth.float(), reduction="mean")
    
    def logcos_loss(self, pred, ground_truth):
        return (pred - ground_truth).cosh().log().mean()
    
    def total_ranking_loss(self, pred, ground_truth):
        # Sign as weight
        pred = pred.unsqueeze(1)
        ground_truth = ground_truth.unsqueeze(1)
        # gt_rank = torch.argsort(torch.argsort(ground_truth, dim=0), dim=0)
        
        score_diff = pred - pred.T

        target = ground_truth - ground_truth.T
        target[target < 1] = 0
        # target[target > 2 * self.margin] = 0
        target = target.sign()

        # weight = 1 / ((gt_rank - gt_rank.T).abs() + 1)
        # loss_mat = F.relu(-(score_diff-self.margin) * target)
        loss_mat = F.relu(-score_diff * target + self.margin)
        # loss_mat = loss_mat * weight
        loss = loss_mat.mean()
        # Rank as weight
        # pred = pred.unsqueeze(1)
        # ground_truth = ground_truth.unsqueeze(1)
        # gt_rank = torch.argsort(torch.argsort(ground_truth, dim=0), dim=0) / ground_truth.size(0)
        # score_diff = pred - pred.T

        # target = ground_truth - ground_truth.T
        # weight = gt_rank - gt_rank.T
        # weight[target <= 1] = 0

        # loss_mat = F.relu(-(score_diff - self.margin) * weight)
        # loss = loss_mat.mean()
        return loss

    def corr_pe2_loss(self, pred, ground_truth):
        return (1 - pearsonr(pred, ground_truth)) ** 2

    def corr_pe_loss(self, pred, ground_truth):
        return -pearsonr(pred, ground_truth)

    def wranknet_loss(self, pred, ground_truth):
        pred = pred.unsqueeze(1)
        gt_rank = (torch.argsort(torch.argsort(ground_truth)) / ground_truth.size(0)).unsqueeze(1)
        gt = ground_truth.unsqueeze(1)

        score_diff = torch.sigmoid(pred - pred.T)
        label_diff = gt - gt.T
        tij = (1.0 + torch.sign(label_diff)) / 2.0

        weight = (gt_rank - gt_rank.T).abs()

        loss_mat = -(tij * torch.log(score_diff) + (1-tij)*torch.log(1-score_diff))
        # loss_mat = loss_mat * weight

        # loss = loss_mat.sum() / weight.sum()
        loss = loss_mat.mean()
        return loss
        