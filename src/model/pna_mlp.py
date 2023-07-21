import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import BaseModel


class PNAMLP(BaseModel, nn.Module):
    """MLP Model for prediction."""
    def __init__(self, args):
        super(PNAMLP, self).__init__(args)
        
        self.mlp = nn.Sequential(
            nn.Linear(args.in_channel * 6, args.in_channel),
            nn.ReLU(),
            nn.Linear(args.in_channel, args.out_channel),
            nn.ReLU(),
            nn.Linear(args.out_channel, 1)
        )
        
    def forward(self, batch):
        x = self.position_encoder(batch)
        x = self.transformer_encoder(x)
        
        x = torch.cat([
            x.mean(dim=1, keepdim=True), 
            x.sum(dim=1, keepdim=True), 
            x.std(dim=1, keepdim=True), 
            x.max(dim=1, keepdim=True).values, 
            x.min(dim=1, keepdim=True).values,
            x[:, [-1], :]
            ], dim=1)

        factors = x[:, -1, :]
        x = self.mlp(x.flatten(1)).squeeze(1)
        return x, factors