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
        embed = self.embed(x)
        for i in range(5):
            embed[:,:,i] += embed[:,:,5]
        embed = embed[:,:,:5,:]
        embed = torch.reshape(embed, (-1, 30,self.embed_dim*5))
        return nn.functional.normalize(embed)