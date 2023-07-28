import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    """Implement a unique loss"""
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        loss_zoo = {
            "default": self.default_loss,
            "mse": self.mse_loss,
        }
        self.loss = loss_zoo[args.loss]

    def forward(self, pred, ground_truth):
        return self.loss(pred, ground_truth)
    
    def mse_loss(self, pred, ground_truth):
        return F.mse_loss(pred.float(), ground_truth.float(), reduction="mean")

    def default_loss(self, pred, ground_truth):
        pass