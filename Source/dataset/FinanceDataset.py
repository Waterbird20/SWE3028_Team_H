import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import FinanceDataReader as fdr


""" Samsumg/Tesla stock data from 2000~2022

    - Samsung
        train/val : 2000~2022
        test : 2022~2023

    -Tesla
        train/val : 2010~2022
        test : 2022~2023

    Author : Chan-young Lee

"""
class FinanceDataset(Dataset):

    def __init__(self, data_args, mode='train'):

        self.stock_id = data_args.stock_id
        self.seq_length = data_args.seq_length
        self.output_length = data_args.output_length

        df = None
        self.X = []
        self.y = []

        if self.stock_id == 'samsung':
            if mode == 'train':
                df = fdr.DataReader('005930','2000','2022')
            elif mode == 'test':
                df = fdr.DataReader('005930','2022','2023')
            else:
                raise ValueError(f'Invalid dataset mode : \"{mode}\"')
            
            df.drop('Change', axis=1)
        
        elif self.stock_id == 'tesla':
            if mode == 'train':
                df = fdr.DataReader('TSLA','2010','2022')
            elif mode == 'test':
                df = fdr.DataReader('TSLA','2022','2023')
            else:
                raise ValueError(f'Invalid dataset mode : \"{mode}\"')
            
            df.drop('Adj Close', axis=1)
            
        else:
            raise ValueError(f'Invalid stock name : {self.stock_id}')
        
        df = df[['Open', 'High', 'Low', 'Volume', 'Close']]
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)

        for i in range(len(df) - self.seq_length - self.output_length):
            self.X.append(df.iloc[i : i+self.seq_length,:])
            self.y.append(df.iloc[i+self.seq_length : i+self.seq_length+self.output_length,-1])
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        assert len(self.X) == len(self.y)

        self.len = len(self.X)


    def __len__(self):
        
        return self.len
    

    def __getitem__(self, idx):

        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        return x,y
        

