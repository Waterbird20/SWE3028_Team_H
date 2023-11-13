import torch
import numpy as np
import torch.nn as nn
from .FinanceEmbedding import FinanceEmbedding
from ..common.FinanceArguments import EmbeddingArgs

class FinanceTransformer(nn.Module):

    def __init__(self, model_args):

        super(FinanceTransformer, self).__init__()

        self.embed_dim = model_args.embed_dim
        self.resolution = model_args.resolution
        self.n_head = model_args.n_head
        self.n_layer = model_args.n_layer
        self.fc_hidden_size = model_args.fc_hidden_size
        self.output_length = model_args.output_length
        
        embed_args = EmbeddingArgs(embed_dim=self.embed_dim, resolution=self.resolution)
        self.embed = FinanceEmbedding(embed_args)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.n_head)
        self.encoder = nn.TransformerEncoder(encoder_layer = self.encoder_layer, num_layers=self.n_layer)
        self.fc1 = nn.Linear(self.embed_dim, self.fc_hidden_size)
        self.fc2 = nn.Linear(self.fc_hidden_size, self.output_length)
        self.act = nn.ELU()

    def forward(self, x):

        embed_x = self.embed(x)
        enc_x = self.encoder(embed_x)
        logits = self.fc1(enc_x)
        logits = self.act(logits)
        logits = self.fc2(logits)
        return logits



