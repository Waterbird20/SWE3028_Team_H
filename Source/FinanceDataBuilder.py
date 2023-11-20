import numpy as np
from sklearn.preprocessing import MinMaxScaler
from common.FinanceArgParser import FinanceParser
import FinanceDataReader as fdr


us_stable = ['KO', 'MCD', 'WM', 'RSG', 'PEP', 'CL', 'WMT', 'CBOE', 'GD', 'KMB', 'PG', 'COR']

us_unstable = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'COST', 'CSCO', 'NFLX']

ko_stable = ['005930', '006400', '033780', '000100', '000660', '005830', '010130', '001450', '138040', '030000', '011070', \
             '004170', '024100', '036570', '058470', '011170', '004370', '012750']

ko_unstable = ['005490', '035420', '005380', '051910', '000270', '068270', '105560', '055550', '373220',  '035720', '012330', \
               '028260',  '207940', '066570']

us_unstable_test = 'AMD'
us_stable_test = 'IBM'

ko_unstable_test = '086790'
ko_stable_test = '081660'

code_list = [us_stable, us_unstable, ko_stable, ko_unstable]
test_list = [us_stable_test, us_unstable_test, ko_stable_test, ko_unstable_test]

filename_list = ['us_stable', 'us_unstable', 'ko_stable', 'ko_unstable']

argparser = FinanceParser('config/databuilder_config.yaml')

args = argparser.parse_databuilder_args()

seq_length = args.seq_length
output_length = args.output_length
predict_type = args.predict_type

scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Volume', 'Close']

predict_index = None

if predict_type == 'high':
    predict_index = 1
elif predict_type == 'low':
    predict_index = 2

print('Build Started')

for i in range(4):

    code_to_load = code_list[i]
    test_code = test_list[i]
    filename = filename_list[i]

    total_seq = []
    total_label = []

    for code in code_to_load:

        print(f'Processing {code}')
        df = fdr.DataReader(code, '2000','2023')
        scaled_array = scaler.fit_transform(df[scale_cols])
        
        for j in range(1, len(df) - seq_length - output_length):
            x = scaled_array[j:j+seq_length, :] - scaled_array[j-1:j+seq_length-1, :]
            y = scaled_array[j+seq_length:j+seq_length+output_length, predict_index] - scaled_array[j+seq_length-1:j+seq_length+output_length-1, predict_index]
            total_seq.append(x)
            total_label.append(y)
    
    total_seq = np.array(total_seq)
    total_label = np.array(total_label)

    assert total_seq.shape[0] == total_label.shape[0]

    np.save('./data/' + filename+'/'+predict_type+'_data.npy', total_seq)
    np.save('./data/' + filename+'/'+predict_type+'_label.npy', total_label)

    df = fdr.DataReader(test_code, '2022')
    scaled_array = scaler.fit_transform(df[scale_cols])
    
    origin = None
    minimum = None
    maximum = None
    if predict_index == 1:
        origin = df['High']
        minimum = min(df['High'])
        maximum = max(df['High'])
    elif predict_index == 2:
        origin = df['Low']
        minimum = min(df['Low'])
        maximum = max(df['Low'])
    
    origin = np.array(origin)
    np.save('./data/' + filename+'/'+predict_type+'_test_origin.npy', origin)

    minmax = np.array([minimum, maximum])
    np.save('./data/' + filename+'/'+predict_type+'_test_minmax.npy', minmax)

    total_seq = []
    total_label = []

    for j in range(1, len(df) - seq_length - output_length):
            x = scaled_array[j:j+seq_length, :] - scaled_array[j-1:j+seq_length-1, :]
            y = scaled_array[j+seq_length:j+seq_length+output_length, predict_index] - scaled_array[j+seq_length-1:j+seq_length+output_length-1, predict_index]
            total_seq.append(x)
            total_label.append(y)
    
    total_seq = np.array(total_seq)
    total_label = np.array(total_label)

    assert total_seq.shape[0] == total_label.shape[0]

    np.save('./data/' + filename+'/'+predict_type+'_test_data.npy', total_seq)
    np.save('./data/' + filename+'/'+predict_type+'_test_label.npy', total_label)



        


