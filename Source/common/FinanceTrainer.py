import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchsummary import summary as summary
from tqdm import tqdm
import matplotlib.pyplot as plt


us_stable = ['KO', 'MCD', 'WM', 'RSG', 'PEP', 'CL', 'WMT', 'CBOE', 'GD', 'KMB', 'PG', 'COR', 'IBM']

us_unstable = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'COST', 'CSCO', 'NFLX', 'AMD']

ko_stable = ['005930', '006400', '033780', '000100', '000660', '005830', '010130', '001450', '138040', '030000', '011070', \
             '004170', '024100', '036570', '058470', '011170', '004370', '012750', '081660']
ko_unstable = ['005490', '035420', '005380', '051910', '000270', '068270', '105560', '055550', '373220',  '035720', '012330', \
               '028260',  '207940', '066570', '086790']


class CustomLoss(_Loss):

    __constants__ = ['alpha', 'epsilon', 'delta', 'reduction']

    def __init__(self, reduction: str = 'mean', alpha: float = 0.5, epsilon: float = 1e-4):
        super().__init__(reduction= reduction)
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor, origin: Tensor) -> Tensor:
        return (3.5/self.alpha)*F.huber_loss(input, target) + self.alpha * torch.mean(torch.sqrt((torch.abs(input-target)/(torch.abs(target-origin) + self.epsilon))**2))


class FinanceTrainer:

    def __init__(self, trainer_args, data_args, model, dataset):


        self.batch_size = trainer_args.batch_size
        self.lr = trainer_args.learning_rate
        self.num_epoch = trainer_args.num_epoch
        self.device = trainer_args.device
        self.resolution = trainer_args.resolution
        self.stock_type = trainer_args.stock_type
        self.dataset = dataset
        self.data_args = data_args
        self.predict_type = data_args.predict_type
        self.save_model = trainer_args.save_model
        self.model_type = trainer_args.model_type

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
        self.model = model.to(self.device)

        self.optim = torch.optim.Adam(model.parameters(), lr = self.lr)

        self.do_train = trainer_args.do_train
        self.do_test = trainer_args.do_test

        if trainer_args.do_train == True:
            self.train_dataset = dataset(data_args, mode='train')
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

            self.lr_sched = torch.optim.lr_scheduler.StepLR(optimizer = self.optim, step_size = int(((len(self.train_dataset) * self.num_epoch)/self.batch_size)/1000), gamma = (0.001)**(0.001))
            #self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim,500)

        self.stock_list = None
        if self.stock_type == 'us_stable':
            self.stock_list = us_stable
        elif self.stock_type == 'us_unstable':
            self.stock_list = us_unstable
    
    def train(self):
        
        print('Train Started')
        print(f'Dataset Length : {len(self.train_dataset)}')
        self.model.train()
        loss_y = []
        loss_acc = 0.

        for i in range(self.num_epoch):
                
            for j, data in tqdm(enumerate(self.train_dataloader), desc=f'Epoch {i}'):
            
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

                if j % 500 == 499:
                    print(f'batch {j+1} loss : {loss.item()}')

        plt.plot(loss_y)
        plt.title("Custom Loss")
        plt.show()

    def test(self):

        print('Test Started')
        self.model.eval()

        self.test_dataset = self.dataset(self.data_args, mode='test', stock_id=self.stock_list[0])
        MPA_list = np.array([0.0 for i in range(len(self.test_dataset)-1)])
        x_graph = np.linspace(0, len(MPA_list)-2,len(MPA_list)-1)

        k=0
        mse = 0.0
        trend = 0
        count = 0
        total_label_y = []
        total_pred_y = []
        for stock_id in self.stock_list:

            i=0 

            label_y = []
            pred_y = []

            self.test_dataset = self.dataset(self.data_args, mode='test', stock_id=stock_id)
            self.test_dataloader = DataLoader(self.test_dataset, shuffle=False)

            for j, data in enumerate(self.test_dataloader):

                if j == 0: continue
                inputs, labels, labels_origin = data
                logits = self.model(inputs.to(self.device))

                label_price = self.test_dataset.inverse_transform(labels)
                pred_price = self.test_dataset.inverse_transform(logits.detach().cpu())
                delta = label_price - pred_price

                if np.sign(label_price) == np.sign(pred_price): trend += 1

                label_y.append(label_price + labels_origin.item())
                pred_y.append(pred_price + labels_origin.item())

                MPA_list[i] += np.average(np.abs(delta)/(label_price+ labels_origin.item()))

                mse += (np.abs(delta)).item()
                count += 1
                i+=1

            total_label_y.append(label_y)
            total_pred_y.append(pred_y)
            k+=1

        MPA_list = 1 - MPA_list/len(self.stock_list)
        x = np.linspace(0, len(MPA_list)-1,len(MPA_list))
        plt.cla()
        plt.plot(x,MPA_list, label='MPA')
        plt.title(f'Average MPA : {np.sum(MPA_list)/len(MPA_list)}\nMAE : {mse/count}\nTrend Accuracy : {trend/count}')
        plt.legend()
        plt.show()

        '''
        x_graph  = np.linspace(0, len(pred_y)-1,len(pred_y))
        plt.cla()
        plt.plot(x_graph,pred_y, label='Predict')
        plt.plot(x_graph,label_y, label='Actual')
        plt.legend()
        plt.show()
        '''
        print(f'Average MPA : {np.sum(MPA_list)/len(MPA_list)}\nMAE : {mse/count}\nTrend Accuracy : {trend/count}')

        np.save(f'./{self.model_type}_{self.stock_type}_{self.predict_type}_graph_actual.npy', np.array(total_label_y))
        np.save(f'./{self.model_type}_{self.stock_type}_{self.predict_type}_graph_predict.npy', np.array(total_pred_y))

        if self.save_model and self.do_train:
            torch.save(self.model.state_dict(), f'savedmodels/{self.model_type}_{self.stock_type}_{self.predict_type}.pth')
            print(f'Model\'s state_dict saved as savedmodels/{self.model_type}_{self.stock_type}_{self.predict_type}.pth')
    
    def start(self):
        if self.do_train == True:
            self.train()
        if self.do_test == True:
            self.test()

