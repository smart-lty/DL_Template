import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import util


class Regularizer(nn.Module):
    """Implement a unique loss"""
    def __init__(self, args):
        super(Regularizer, self).__init__()
        self.args = args
        self.reg = args.reg
    
    def forward(self, features, ground_truth, num_samples):
        arg_rank = torch.argsort(ground_truth, descending=True)
        # indexes = torch.randperm(arg_rank.size(0))
        top_size = arg_rank.size(0) // 4

        anchor = arg_rank[:200]

        pos_set = arg_rank[:top_size]
        neg_set = arg_rank[top_size:]

        pos_samples = pos_set[torch.randint(low=0, high=pos_set.size(0), size=(num_samples,))]
        neg_samples = neg_set[torch.randint(low=0, high=neg_set.size(0), size=(num_samples,))]

        anc_features = features[anchor].mean(dim=0, keepdim=True)
        pos_features = features[pos_samples]
        neg_features = features[neg_samples]

        loss = F.relu((anc_features - pos_features).norm(dim=1) - (anc_features - neg_features).norm(dim=1) + self.args.margin).mean()
        loss *= self.reg
        return loss


