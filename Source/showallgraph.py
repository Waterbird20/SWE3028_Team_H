import numpy as np
from matplotlib import pyplot as plt

us_stable = ['KO', 'MCD', 'WM', 'RSG', 'PEP', 'CL', 'WMT', 'CBOE', 'GD', 'KMB', 'PG', 'COR', 'IBM']

us_unstable = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'COST', 'CSCO', 'NFLX', 'AMD']

model_type = ['LSTM', 'GRU', 'TF']
stock_type = ['us_stable', 'us_unstable']
predict_type = ['high', 'low']

def show_graph(a, p, i, mt, st, pt):
    x = np.linspace(0, len(a)-1, len(a))
    plt.cla()
    if st == 'us_unstable':
        plt.title(f'{mt}_{st}_{pt} : {us_unstable[i]}')
    else:
        plt.title(f'{mt}_{st}_{pt} : {us_stable[i]}')
    plt.plot(x,p, label='Predict')
    plt.plot(x,a, label='Actual')
    plt.legend()
    plt.show()


for mt in model_type:
    for st in stock_type:
        for pt in predict_type:
            a = np.load(f'./{mt}_{st}_{pt}_graph_actual.npy')
            p = np.load(f'./{mt}_{st}_{pt}_graph_predict.npy')

            for i in range(len(a)):
                show_graph(a[i], p[i], i, mt, st, pt)

