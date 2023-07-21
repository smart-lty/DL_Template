import torch
import torch.nn as nn
import torch.nn.functional as F
from model.positional_encoding import BERTPositionalEncoding, TransformerPositionalEncoding


class BaseModel(nn.Module):
    """Base Model for prediction."""
    def __init__(self, args):
        super(BaseModel, self).__init__()
        
        self.args = args

        self.in_channel = args.in_channel
        self.out_channel = args.out_channel

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=args.in_channel, nhead=args.nheads, dim_feedforward=args.in_channel,dropout=args.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=args.num_layers)
        
        if args.pe == "bert":
            self.position_encoder = BERTPositionalEncoding(d_model=args.in_channel, max_len=20)
        elif args.pe == "transformer":
            self.position_encoder = TransformerPositionalEncoding(d_model=args.in_channel, max_len=20)
    
    def forward(self, batch):
        pass