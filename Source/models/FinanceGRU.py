import torch
from torch import nn

class FinanceGRU(nn.Module):

    def __init__(self, model_args):

        super(FinanceGRU, self).__init__()

        self.output_length = model_args.output_length
        self.num_layers = model_args.num_layers
        self.input_size = model_args.input_size
        self.hidden_size = model_args.hidden_size
        self.fc_hidden_size = model_args.fc_hidden_size
        self.device = model_args.device

        self.gru = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size,
                            num_layers = self.num_layers, batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, self.fc_hidden_size)
        self.fc2 = nn.Linear(self.fc_hidden_size, self.output_length)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        h_0 = torch.Tensor(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        output, (hn) = self.gru(x, (h_0))
        hn = hn.view(-1, self.hidden_size)
        logits = self.relu(hn)
        logits = self.fc1(logits)
        logits = self.relu(logits)
        logits = self.fc2(logits)

        return logits

