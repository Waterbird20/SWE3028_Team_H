import FinanceDataReader as fdr
import pandas as pd

us_stable = ['KO', 'MCD', 'WM', 'RSG', 'PEP', 'CL', 'WMT', 'CBOE', 'GD', 'KMB', 'PG', 'COR', 'IBM']

us_unstable = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'COST', 'CSCO', 'NFLX', 'AMD']

ko_stable = ['005930', '006400', '033780', '000100', '000660', '005830', '010130', '001450', '138040', '030000', '011070', \
             '004170', '024100', '036570', '058470', '011170', '004370', '012750', '081660']
ko_unstable = ['005490', '035420', '005380', '051910', '000270', '068270', '105560', '055550', '373220',  '035720', '012330', \
               '028260',  '207940', '066570', '086790']

date = '2023-12-11'

class FinanceTrainer:

    def __init__(self, trainer_args, data_args, model_high, model_low, dataset):


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

        self.test_dataset = None
        self.test_dataloader = None

        self.model_high = model_high.to(self.device)
        self.model_low = model_low.to(self.device)

        self.stock_list = None
        if self.stock_type == 'us_stable':
            self.stock_list = us_stable
        elif self.stock_type == 'us_unstable':
            self.stock_list = us_unstable

    def test(self):

        print('Upload Started')
        self.model_high.eval()
        self.model_low.eval()
        i=0
        for stock_id in self.stock_list:
            self.origin_csv = fdr.DataReader(stock_id, '2023')
            self.origin_csv = self.origin_csv[['Open', 'High', 'Low', 'Close']]
            self.data_args.predict_type = 'high'
            self.test_dataset_high = self.dataset(self.data_args, mode='test', stock_id=self.stock_list[i])
            self.data_args.predict_type = 'low'
            self.test_dataset_low = self.dataset(self.data_args, mode='test', stock_id=self.stock_list[i])

            inputs, labels_origin = self.test_dataset_high[0]
            logits = self.model_high(inputs.to(self.device))

            pred_price_high = labels_origin.item() + self.test_dataset_high.inverse_transform(logits.detach().cpu()).item()

            inputs, labels_origin = self.test_dataset_low[0]
            logits = self.model_low(inputs.to(self.device))
            pred_price_low = labels_origin.item() + self.test_dataset_low.inverse_transform(logits.detach().cpu()).item()

            print(pred_price_high, pred_price_low)
            new_data = {'Open':None, 'High': pred_price_high, 'Low': pred_price_low, 'Close':None}
            new_ts = pd.Timestamp(date)
            new_column = pd.DataFrame(new_data, index=[new_ts])

            self.origin_csv = pd.concat([self.origin_csv, new_column])

            self.origin_csv.to_csv(f'./uploadtest/{stock_id}_{self.model_type}.csv')
            i += 1

    
    def start(self):
        self.test()

