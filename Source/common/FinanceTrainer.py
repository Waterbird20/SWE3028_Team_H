import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchsummary import summary as summary
import matplotlib.pyplot as plt

class CustomLoss(_Loss):

    __constants__ = ['alpha', 'epsilon', 'delta', 'reduction']

    def __init__(self, reduction: str = 'mean', alpha: float = 0.1, epsilon: float = 1e-4, delta: float = 1.0):
        super().__init__(reduction= reduction)
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor, origin: Tensor) -> Tensor:
        return self.alpha * torch.mean(torch.sqrt((torch.abs(input-target)/(torch.abs(target-origin) + self.epsilon))**2))


class FinanceTrainer:

    def __init__(self, trainer_args, data_args, model, dataset):


        self.batch_size = trainer_args.batch_size
        self.lr = trainer_args.learning_rate
        self.num_epoch = trainer_args.num_epoch
        self.predict_index = None
        if data_args.predict_type == 'high':
            self.predict_index = 1
        elif data_args.predict_type == 'low':
            self.predict_index = 2

        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None

        self.loss_fn = CustomLoss()
        self.device = trainer_args.device
        self.model = model.to(self.device)

        self.optim = torch.optim.Adam(model.parameters(), lr = self.lr)

        self.do_train = trainer_args.do_train
        self.do_test = trainer_args.do_test

        if trainer_args.do_train == True:
            self.train_dataset = dataset(data_args, mode='train')
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        if trainer_args.do_test == True:
            self.test_dataset = dataset(data_args, mode='test')
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def train(self):
        
        self.model.train()
        loss_y = []
        loss_acc = 0.
        for i in range(self.num_epoch):
            
            print(f'Epoch {i}:')
            for j, data in enumerate(self.train_dataloader):
             
                inputs, labels = data
                self.optim.zero_grad()
                
                logits = self.model(inputs.to(self.device))
                loss = self.loss_fn(logits, labels.to(self.device), inputs[:,-1,1].unsqueeze(-1).to(self.device))
                if j % 50 == 49:
                    loss_y.append(loss_acc/50)
                    loss_acc = 0.0
                else:
                    loss_acc += loss.item()
                loss.backward()

                self.optim.step()

                if j % 100 == 99:
                    print(f'batch {j+1} loss : {loss.item()}')
        plt.plot(loss_y)
        plt.title("Custom Loss")
        plt.show()

    def test(self):

        self.model.eval()
        avg_delta = 0.
        x = np.linspace(1, len(self.test_dataloader)-1,len(self.test_dataloader)-1)
        label_y = []
        pred_y = []
        for j, data in enumerate(self.test_dataloader):

            if j == 0: continue
            inputs, labels, labels_origin = data
            logits = self.model(inputs.to(self.device))

            label_price = self.test_dataset.inverse_transform(labels)
            pred_price = self.test_dataset.inverse_transform(logits.detach().cpu())
            label_y.append(label_price + labels_origin.item())
            pred_y.append(pred_price + labels_origin.item())
            avg_delta += np.average(np.abs(pred_price - label_price))

        plt.cla()
        plt.plot(x,label_y, label='Actual')
        plt.plot(x,pred_y, label='Predict')
        plt.legend()
        plt.show()
        print(f'MAE in KRW = {avg_delta/len(self.test_dataloader)}')
    
    def start(self):
        if self.do_train == True:
            print(f'Training Started')
            self.train()
        if self.do_test == True:
            print(f'Testing')
            self.test()

