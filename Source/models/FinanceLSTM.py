import torch
import torch.nn as nn

class FinanceLSTM(nn.Module):

    def __init__(self, model_args):

        super(FinanceLSTM, self).__init__()

        self.output_length = model_args.output_length
        self.num_layers = model_args.num_layers
        self.input_size = model_args.input_size
        self.hidden_size = model_args.hidden_size
        self.fc_hidden_size = model_args.fc_hidden_size
        self.device = model_args.device
        
        self.net = nn.ModuleList()
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,
                            num_layers = self.num_layers, batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, self.fc_hidden_size)
        self.fc2 = nn.Linear(self.fc_hidden_size, self.output_length)
        self.act = nn.ELU()
    
    def forward(self, x):

        h_0 = torch.Tensor(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        c_0 = torch.Tensor(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        logits = self.act(hn)
        logits = self.fc1(logits)
        logits = self.act(logits)
        logits = self.fc2(logits)
        return logits
