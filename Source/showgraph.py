import numpy as np
from matplotlib import pyplot as plt

us_stable = ['KO', 'MCD', 'WM', 'RSG', 'PEP', 'CL', 'WMT', 'CBOE', 'GD', 'KMB', 'PG', 'COR', 'IBM']

us_unstable = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'COST', 'CSCO', 'NFLX', 'AMD']

def show_graph(a, p, i):
    x = np.linspace(0, len(a)-1, len(a))
    plt.cla()
    plt.title(us_stable[i])
    plt.plot(x,p, label='Predict')
    plt.plot(x,a, label='Actual')
    plt.legend()
    plt.show()

a = np.load('./TF_us_stable_high_graph_actual.npy')
p = np.load('./TF_us_stable_high_graph_predict.npy')

for i in range(len(a)):
    show_graph(a[i], p[i], i)

