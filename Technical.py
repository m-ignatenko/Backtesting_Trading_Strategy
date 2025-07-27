import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime 
import seaborn as sns
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
class Backtest(object):
    def __init__(self,symbol, start,end,amount,ftc,ptc,verbose ):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_amount = amount 
        self.amount = amount # starting balance value
        self.ftc = ftc # fixed transaction cost per 1 trade
        self.ptc = ptc # proportional transaction costs per 1 trade
        self.units = 0 # number of shares in portfolio initially
        self.pos = 0 # market neutral initial position
        self.trades = 0 # number of trades 
        self.verbose = verbose # True to get full output
        self.buys = {}
        self.sells = {}
        self.msg = ""
    def get_data(self):
        data = pd.DataFrame(yf.download(self.symbol, self.start, self.end))['Close']
        data.rename(columns={self.symbol:'price'},inplace=True)
        data['return'] = np.log(data['price'] / data['price'].shift(1))
        self.data = data.dropna()
    def plot(self):
        self.data['price'].plot(figsize=(10,6),label='price')
        plt.title("Price and signals")
        plt.plot(self.buys.keys(),self.buys.values(),'go',label='buy',markersize=5)
        plt.plot(self.sells.keys(),self.sells.values(),'ro',label='sell',markersize=5)
        plt.legend()
        plt.show()
    def get_price(self, day):
        return self.data['price'].iloc[day]
    def get_wealth(self, day):
        return self.data['price'].iloc[day]*self.units + self.amount
    def buy_order(self,day, buy_units=None, buy_amount=None):
        price = self.get_price(day)
        if buy_units == None:
            buy_units = int(buy_amount/price)
        self.amount -= (buy_units * price)*(1 + self.ptc) + self.ftc
        self.units += buy_units
        self.trades += 1
        self.buys[self.data.index[day]] = price
        if self.verbose == True:
            self.msg += f'{str(self.data.index[day])[:10]}: buying {buy_units} units at {price:.2f}\n'
            # print(f"wealth: {self.get_wealth(day):.2f}, balance: {self.amount:.2f}, ")
    def sell_order(self,day, sell_units=None, sell_amount=None):
        price = self.get_price(day)
        if sell_units == None:
            sell_units = int(sell_amount/price)
        self.amount += (sell_units * price)*(1 - self.ptc) - self.ftc
        self.units -= sell_units
        self.trades += 1
        self.sells[self.data.index[day]] = price
        if self.verbose == True:
            self.msg +=  f'{str(self.data.index[day])[:10]}: selling {sell_units} units at {price:.2f}\n'
            # print(f"wealth: {self.get_wealth(day):.2f}, balance: {self.amount:.2f}, units: {self.units}")
    def close_postion(self, day):
        price = self.get_price(day)
        self.amount += self.units * price
        self.units = 0
        self.trades +=1 
        # print('=' * 55)
        if self.verbose == True:
            self.msg += f"Total wealth: {self.get_wealth(day):.2f}\n"
        perf = (self.amount - self.initial_amount)/self.initial_amount * 100
        # print(f'Net Performance [%] {perf:.2f}') # net performance in %
        # print(f'Trades Executed [#] {self.trades:.2f}') 
        # print('=' * 55)
    # Strategies:
    def sma_crossover(self, sma1, sma2):
        if sma2 < sma1:
            sma1, sma2 = sma2, sma1
        self.data['SMA1'] = self.data['price'].rolling(sma1).mean()
        self.data['SMA2'] = self.data['price'].rolling(sma2).mean()
        self.pos = 0 #neutral
        for i in range(sma2, self.data.shape[0]):
            if self.pos == 0 and self.data['SMA1'].iloc[i] > self.data['SMA2'].iloc[i]:
                self.buy_order(i,buy_amount=self.amount)
                self.trades+=1
                self.pos = 1
            elif self.pos == 1 and self.data['SMA1'].iloc[i] < self.data['SMA2'].iloc[i]:
                self.sell_order(i,sell_units=self.units)
                self.trades+=1
                self.pos = 0
        self.close_postion(i)

    def momentum(self, sma):
        self.data['SMA'] = self.data['return'].rolling(sma).mean()
        self.pos = 0
        for i in range(sma, self.data.shape[0]):
            if 0 < self.data['SMA'].iloc[i] and self.pos == 0:
                self.buy_order(i,buy_amount=self.amount)
                self.pos = 1
                self.trades+=1
            elif 0 > self.data['SMA'].iloc[i] and self.pos == 1:
                self.sell_order(i,sell_units=self.units)
                self.pos = 0
                self.trades +=1
        self.close_postion(i)
    def mean_reversed(self, sma,th):
        self.data['SMA'] = self.data['price'].rolling(sma).mean()
        self.pos = 0
        for i in range(sma, self.data.shape[0]):
            if (self.data['price'].iloc[i] <= self.data['SMA'].iloc[i] - th) and self.pos == 0:
                self.buy_order(i,buy_amount=self.amount)
                self.pos = 1
                self.trades+=1
            elif (self.data['price'].iloc[i] >self.data['SMA'].iloc[i] + th) and self.pos == 1:
                self.sell_order(i,sell_units=self.units)
                self.pos = 0
                self.trades +=1
        self.close_postion(i)
    