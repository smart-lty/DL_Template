import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel


class MLP(BaseModel, nn.Module):
    """MLP Model for prediction."""
    def __init__(self, args):
        super(MLP, self).__init__(args)
        
        self.mlp = nn.Sequential(
            nn.Linear(args.in_channel, args.out_channel),
            nn.ReLU(),
            nn.Linear(args.out_channel, 1)
        )
        
    def forward(self, batch):
        x = self.position_encoder(batch)
        x = self.transformer_encoder(x)
        factors = x[:, -1, :]
        x = self.mlp(factors).squeeze(1)
        return x, factors