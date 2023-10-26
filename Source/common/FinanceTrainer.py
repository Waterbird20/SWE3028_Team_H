import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary as summary

class FinanceTrainer:

    def __init__(self, trainer_args, data_args, model, dataset):


        self.batch_size = trainer_args.batch_size
        self.lr = trainer_args.learning_rate
        self.num_epoch = trainer_args.num_epoch
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None

        self.loss_fn = nn.HuberLoss()
        self.model = model

        self.optim = torch.optim.Adam(model.parameters(), lr = self.lr)

        if trainer_args.do_train == True:
            self.train_dataset = dataset(data_args, mode='train')
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        if trainer_args.do_test == True:
            self.test_dataset = dataset(data_args, mode='test')
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)

    def train(self):
        
        self.model.train()
        for i in range(self.num_epoch):
            
            print(f'Epoch {i}:')
            for j, data in enumerate(self.train_dataloader):
                
                inputs, labels = data
                self.optim.zero_grad()
                
                logits = self.model(inputs)
                loss = self.loss_fn(logits, labels)
                loss.backward()

                self.optim.step()

                if j % 100 == 99:
                    print(f'batch {j+1} loss : {loss.item()}')

    def test(self):

        self.model.eval()
        avg_delta = 0.
        for j, data in enumerate(self.test_dataloader):

            inputs, labels = data
            logits = self.model(inputs)
            label_price = self.test_dataset.inverse_transform(labels)
            pred_price = self.test_dataset.inverse_transform(logits.detach())
            avg_delta += np.average(np.abs(pred_price - label_price))

        print(f'Avg delta = {avg_delta/len(self.test_dataloader)}')

