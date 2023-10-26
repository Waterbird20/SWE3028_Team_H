import torch
import numpy as np
import pandas as pd
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
        self.predict_type = data_args.predict_type

        df = None
        self.X = []
        self.y = []
        
        self.min = None
        self.max = None

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
        
        
        predict_index = None

        if self.predict_type == 'high':
            predict_index = 1
            self.min = min(df['High'])
            self.max = max(df['High'])

        elif self.predict_type == 'low':
            predict_index = 2
            self.min = min(df['Low'])
            self.max = max(df['Low'])

        else:
            raise ValueError(f'Invalid predict type : \"{self.predict_type}\"')
        

        scaler = MinMaxScaler()
        df = df[['Open', 'High', 'Low', 'Volume', 'Close']]
        scale_cols = ['Open', 'High', 'Low', 'Volume', 'Close']
        scaled_array = scaler.fit_transform(df[scale_cols])
        df = pd.DataFrame(scaled_array, columns = scale_cols)
        

        for i in range(len(df) - self.seq_length - self.output_length):
            x = np.array(df.iloc[i : i+self.seq_length,:])
            y = np.array(df.iloc[i+self.seq_length : i+self.seq_length+self.output_length, predict_index])
            self.X.append(x)
            self.y.append(y)
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        print(self.X.shape, self.y.shape)
        assert len(self.X) == len(self.y)

        self.len = len(self.X)


    def inverse_transform(self, x):
        return np.array(self.min + (self.max - self.min) * x).squeeze(0)

    def __len__(self):     
        return self.len
    

    def __getitem__(self, idx):

        x = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        return x,y
        

