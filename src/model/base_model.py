import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """Base Model for prediction."""
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
    
    def forward(self, batch):
        pass