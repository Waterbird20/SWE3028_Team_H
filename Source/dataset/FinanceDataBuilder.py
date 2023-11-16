import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import FinanceDataReader as fdr

us_stable = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'COST', 'CSCO', 'NFLX', 'AMD']
us_unstable = ['KO', 'MCD', 'WM', 'RSG', 'PEP', 'CL', 'WMT', 'CBOE', 'GD', 'KMB', 'PG', 'COR', 'IBM']

ko_stable = ['005930', '006400', '033780', '000100', '000660', '005830', '010130', '001450', '138040', '030000', '011070', \
             '004170', '024100', '036570', '058470', '011170', '004370', '012750', '081660']
ko_unstable = ['005490', '035420', '005380', '051910', '000270', '068270', '105560', '055550', '373220',  '035720', '012330', \
               '028260',  '207940', '066570', '086790']

code_list = [us_stable, us_unstable, ko_stable, ko_unstable]
filename_list = []