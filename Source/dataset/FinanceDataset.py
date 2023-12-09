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

us_stable = ['KO', 'MCD', 'WM', 'RSG', 'PEP', 'CL', 'WMT', 'CBOE', 'GD', 'KMB', 'PG', 'COR', 'IBM']

us_unstable = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'COST', 'CSCO', 'NFLX', 'AMD']

ko_stable = ['005930', '006400', '033780', '000100', '000660', '005830', '010130', '001450', '138040', '030000', '011070', \
             '004170', '024100', '036570', '058470', '011170', '004370', '012750', '081660']
ko_unstable = ['005490', '035420', '005380', '051910', '000270', '068270', '105560', '055550', '373220',  '035720', '012330', \
               '028260',  '207940', '066570', '086790']

class FinanceDataset(Dataset):

    def __init__(self, data_args, mode='train', stock_id=None):

        self.data_path = data_args.data_path
        self.seq_length = data_args.seq_length
        self.output_length = data_args.output_length
        self.predict_type = data_args.predict_type
        self.stock_id = stock_id
        self.mode = mode
        self.is_upload = data_args.is_upload

        self.df = None
        self.X = []
        self.y = []
        
        self.min = None
        self.max = None
        self.origin = None

        if self.is_upload:
            self.X = np.load(self.data_path+'/'+self.stock_id+self.predict_type+'_test_data.npy')
            minmax = np.load(self.data_path+'/'+self.stock_id+self.predict_type+'_test_minmax.npy')
            self.origin = np.load(self.data_path+'/'+self.stock_id+self.predict_type+'_test_origin.npy')
            self.min = minmax[0]
            self.max = minmax[1]

        elif self.mode == 'train':
            self.X = np.load(self.data_path+'/'+self.predict_type+'_data.npy')
            self.y = np.load(self.data_path+'/'+self.predict_type+'_label.npy')

        elif self.mode == 'test':
            self.X = np.load(self.data_path+'/test_'+self.predict_type+'/'+self.stock_id+self.predict_type+'_test_data.npy')
            self.y = np.load(self.data_path+'/test_'+self.predict_type+'/'+self.stock_id+self.predict_type+'_test_label.npy')
            minmax = np.load(self.data_path+'/test_'+self.predict_type+'/'+self.stock_id+self.predict_type+'_test_minmax.npy')
            self.origin = np.load(self.data_path+'/test_'+self.predict_type+'/'+self.stock_id+self.predict_type+'_test_origin.npy')

            self.min = minmax[0]
            self.max = minmax[1]

        if not self.is_upload:
            assert len(self.X) == len(self.y)
        else:
            self.X = np.array([self.X])
        self.len = len(self.X)


    def inverse_transform(self, x):
        return np.array((self.max - self.min) * x).squeeze(0)

    def __len__(self):     
        return self.len
    

    def __getitem__(self, idx):
        if self.is_upload:
            x = torch.FloatTensor(self.X[idx])
            y_origin = torch.FloatTensor(np.array([self.origin[idx+self.seq_length]]))
            return x, y_origin
        if self.mode == 'train':
            x = torch.FloatTensor(self.X[idx])
            y = torch.FloatTensor(self.y[idx])
        elif self.mode == 'test':
            x = torch.FloatTensor(self.X[idx])
            y = torch.FloatTensor(self.y[idx])
            y_origin = torch.FloatTensor(np.array([self.origin[idx+self.seq_length]]))
            return x, y, y_origin
        return x,y
        

