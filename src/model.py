import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import math


class Model(nn.Module):
    """Implement Transformer network to forecast stock prices. (regression task)"""
    def __init__(self, args):
        super(Model, self).__init__()
        self.in_channel = args.in_channel
        self.out_channel = args.out_channel

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=args.in_channel, nhead=args.nheads, dim_feedforward=args.in_channel,dropout=args.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=args.num_layers)
        
        if args.pe == "bert":
            self.position_encoder = BERTPositionalEncoding(d_model=args.in_channel, dropout=0, max_len=20)
        elif args.pe == "transformer":
            self.position_encoder = TransformerPositionalEncoding(d_model=args.in_channel, dropout=0, max_len=20)
        
        # self.compare_layer = nn.TransformerEncoderLayer(d_model=args.in_channel, nhead=args.nheads, dim_feedforward=args.in_channel,dropout=args.dropout)
        # self.compare_encoder = nn.TransformerEncoder(self.compare_layer, num_layers=args.num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(args.in_channel, args.out_channel),
            nn.ReLU(),
            nn.Linear(args.out_channel, 1)
        )
    
    def forward(self, batch):
        """using transformer block to predict"""
        # x = self.position_encoder(batch)

        # x = self.compare_encoder(x)
        # # x = batch
        # x = self.transformer_encoder(x)
        # x = self.mlp(x)[:, -1, :]

        x = self.position_encoder(batch)
        # x = batch
        x = self.transformer_encoder(x)
        features = x[:, -1, :]
        # x = self.compare_encoder(x).squeeze(1)
        x = self.mlp(features)
        return x, features


class PNAModel(nn.Module):
    """
    Implement Transformer network to forecast stock prices. (regression task)
    * Add PNA algorithm.
    """
    def __init__(self, args):
        super(PNAModel, self).__init__()
        self.in_channel = args.in_channel
        self.out_channel = args.out_channel

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=args.in_channel, nhead=args.nheads, dim_feedforward=args.in_channel,dropout=args.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=args.num_layers)
        
        if args.pe == "bert":
            self.position_encoder = BERTPositionalEncoding(d_model=args.in_channel, dropout=0, max_len=20)
        elif args.pe == "transformer":
            self.position_encoder = TransformerPositionalEncoding(d_model=args.in_channel, dropout=0, max_len=20)
        
        # self.compare_layer = nn.TransformerEncoderLayer(d_model=args.in_channel, nhead=args.nheads, dim_feedforward=args.in_channel,dropout=args.dropout)
        # self.compare_encoder = nn.TransformerEncoder(self.compare_layer, num_layers=args.num_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(args.in_channel * 6, args.in_channel),
            nn.ReLU(),
            nn.Linear(args.in_channel, args.out_channel),
            nn.ReLU(),
            nn.Linear(args.out_channel, 1)
        )
    
    def forward(self, batch):
        """using transformer block to predict"""

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
        
        features = x[:, -1, :]
        # x = self.compare_encoder(x).squeeze(1)
        
        x = self.mlp(x.flatten(1))
        return x, features


class BERTPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # x = x + self.pe[:x.size(1)].squeeze(1).unsqueeze(0)
        x = x + self.pe.weight.data.unsqueeze(0)
        return x


class TransformerPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # self.pe = nn.Embedding(max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)].squeeze(1).unsqueeze(0)
        return x