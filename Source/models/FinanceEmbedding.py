import torch
import math
import torch.nn as nn

class FinanceEmbedding(nn.Module):

    def __init__(self, embed_args):

        super(FinanceEmbedding, self).__init__()

        self.embed_dim = embed_args.embed_dim
        self.resolution = embed_args.resolution

        self.embed = nn.Embedding(self.resolution, self.embed_dim)

    '''
    def resolution_map(self, input):
        return torch.LongTensor(list(map(lambda x : torch.round((x+1.0-1/(2*(self.resolution-1)))*(self.resolution-1)/2.0).long(), input)))
    '''

    def forward(self, x):
        return nn.functional.normalize(self.embed(x))