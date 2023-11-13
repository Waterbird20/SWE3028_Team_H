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
        self.mode = mode

        self.df = None
        self.X = []
        self.y = []
        
        self.min = None
        self.max = None

        if self.stock_id == 'samsung':
            if mode == 'train':
                self.df = fdr.DataReader('005930','2000','2022')
            elif mode == 'test':
                self.df = fdr.DataReader('005930','2022','2023')
            else:
                raise ValueError(f'Invalid dataset mode : \"{mode}\"')
            
            self.df.drop('Change', axis=1)
        
        elif self.stock_id == 'tesla':
            if mode == 'train':
                self.df = fdr.DataReader('TSLA','2010','2022')
            elif mode == 'test':
                self.df = fdr.DataReader('TSLA','2022','2023')
            else:
                raise ValueError(f'Invalid dataset mode : \"{mode}\"')
            
            self.df.drop('Adj Close', axis=1)
            
        else:
            raise ValueError(f'Invalid stock name : {self.stock_id}')
        
        
        predict_index = None

        if self.predict_type == 'high':
            predict_index = 1
            self.min = min(self.df['High'])
            self.max = max(self.df['High'])

        elif self.predict_type == 'low':
            predict_index = 2
            self.min = min(self.df['Low'])
            self.max = max(self.df['Low'])

        else:
            raise ValueError(f'Invalid predict type : \"{self.predict_type}\"')
        
        self.predict_index = predict_index

        scaler = MinMaxScaler()
        self.df_origin = None
        self.df = self.df[['Open', 'High', 'Low', 'Volume', 'Close']]
        if predict_index == 1:
            self.df_origin = self.df['High']
        else:
            self.df_origin = self.df['Low']
        scale_cols = ['Open', 'High', 'Low', 'Volume', 'Close']
        scaled_array = scaler.fit_transform(self.df[scale_cols])
        self.df = pd.DataFrame(scaled_array, columns = scale_cols)
        

        for i in range(1, len(self.df) - self.seq_length - self.output_length):
            x = np.array(self.df.iloc[i : i+self.seq_length,:]) - np.array(self.df.iloc[i-1 : i+self.seq_length-1,:])
            y = np.array(self.df.iloc[i+self.seq_length : i+self.seq_length+self.output_length, predict_index]) - np.array(self.df.iloc[i+self.seq_length-1 : i+self.seq_length+self.output_length-1, predict_index])
            self.X.append(x)
            self.y.append(y)
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        assert len(self.X) == len(self.y)

        self.len = len(self.X)


    def inverse_transform(self, x):
        return np.array((self.max - self.min) * x).squeeze(0)

    def __len__(self):     
        return self.len
    

    def __getitem__(self, idx):

        if self.mode == 'train':
            x = torch.FloatTensor(self.X[idx])
            y = torch.FloatTensor(self.y[idx])
        elif self.mode == 'test':
            x = torch.FloatTensor(self.X[idx])
            y = torch.FloatTensor(self.y[idx])
            y_origin = torch.FloatTensor(np.array([self.df_origin.iloc[idx+self.seq_length]]))
            return x, y, y_origin
        return x,y
        

