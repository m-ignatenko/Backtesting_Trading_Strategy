import yfinance as yf
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime as dt
import seaborn as sns   

class Strategy(object):
    def __init__(self, ticker, train_start, train_end, test_start, test_end, transaction_cost, capital):
        self.ticker = ticker
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.transaction_cost = transaction_cost
        self.capital = capital
        self.prepare_data()
    def prepare_data(self):
        df_train= yf.download(self.ticker,start=self.train_start, end=self.train_end)['Close']
        df_test= yf.download(self.ticker,start=self.test_start, end=self.test_end)['Close']
        df_test.rename(columns={self.ticker:"price"},inplace=True)
        df_train.rename(columns={self.ticker:"price"},inplace=True)
        df_train['return'] = np.log(df_train['price']/df_train['price'].shift(1))
        df_test['return'] = np.log(df_test['price']/df_test['price'].shift(1))
        lags =5
        cols = []

        for i in range(1, lags+1):
            col = f'lag_{i}'
            df_train[col] = df_train['return'].shift(i)
            df_test[col] = df_test['return'].shift(i)
            cols.append(col)
        self.cols = cols
        df_train['momentum'] = df_train['return'].rolling(10).mean().shift(1)
        df_test['momentum'] = df_test['return'].rolling(10).mean().shift(1)
        df_train['volatility'] = df_train['return'].rolling(20).std().shift(1)
        df_test['volatility'] = df_test['return'].rolling(20).std().shift(1)
        df_train.dropna(inplace=True)
        df_test.dropna(inplace=True)
        self.df_train = df_train
        self.df_test = df_test
    def model_selection(self, type = 'logregression'):
        if type == 'logregression': 
            self.model = lm.LogisticRegression(C=1e7, solver='lbfgs', multi_class='auto',max_iter=1000)
            self.model.fit(self.df_train[self.cols], np.sign(self.df_train['return']))
            self.df_train['prediction']= self.model.predict(self.df_train[self.cols])
            self.df_test['prediction']= self.model.predict(self.df_test[self.cols])
            self.df_train['strategy'] = self.df_train['prediction'] *self.df_train['return']
            self.df_test['strategy'] = self.df_test['prediction'] *self.df_test['return']
            self.df_train.dropna(inplace=True)
            self.df_test.dropna(inplace=True)

    def estimation(self):
        self.model_selection()
        # ON TRAIN DATA
        sns.set_theme(
        style="darkgrid",
        palette="deep",
        font_scale=1.1,
        rc={
            'figure.figsize': (10, 6),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.edgecolor': '0.15',
            'axes.linewidth': 1.25,
        }
    )
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # 1 row, 2 columns
        gs = gridspec.GridSpec(1, 2)
        ax1.set_title("Train data calculation")

        ax1.plot(self.df_train['return'].cumsum().apply(np.exp), label='basic return',c='b')
        ax1.plot(self.df_train['strategy'].cumsum().apply(np.exp), label='strategy',color='purple')
        a = self.df_train['return'].cumsum().apply(np.exp)[-1]
        b = self.df_train['strategy'].cumsum().apply(np.exp)[-1]
        data = dt.datetime.strptime(self.train_end, '%Y-%m-%d')
        if (a <= b):
            ax1.vlines(data,ymin=min(a,b),ymax= max(a,b),colors='g')
        else:
            ax1.vlines(data,ymin=min(a,b),ymax= max(a,b),colors='red')
        ax1.legend(loc=0)

        ax2.set_title("Test data calculation")

        ax2.plot(self.df_test['return'].cumsum().apply(np.exp), label='basic return',c='b')
        ax2.plot(self.df_test['strategy'].cumsum().apply(np.exp), label='strategy',color='purple')
        a = self.df_test['return'].cumsum().apply(np.exp)[-1]
        b = self.df_test['strategy'].cumsum().apply(np.exp)[-1]
        
        data = dt.datetime.strptime(self.test_end, '%Y-%m-%d')
        
        if (a <= b):
            ax2.vlines(data,ymin=min(a,b),ymax= max(a,b),colors='g')
        else:
            ax2.vlines(data,ymin=min(a,b),ymax= max(a,b),colors='red')
        fig.subplots_adjust(top=0.85)
        ax2.legend(loc=0)
        plt.show()
    def run(self):
        self.prepare_data()
        self.estimation()