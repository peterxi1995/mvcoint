from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ecm_coint import *
from collections import deque
from itertools import combinations



class Strategy:
    def __init__(self,assetNames):
        #parameters

        self.assetNames = assetNames
        self.cointLookBack = 250
        self.capital = 1
        self.maturity = 1
        self.timeLeft = 0
        self.assetHistClose = {}
        self.var = None
        self.current_asset = 1
        for asset in self.assetNames:
            self.assetHistClose[asset] = deque([],maxlen=self.cointLookBack)

    def fit_var(self):
        self.var = None
        if len(self.assetHistClose[self.assetNames[0]]) == self.cointLookBack:
            print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            print 'Fitting var....'
            data = pd.DataFrame(self.assetHistClose)
            data = data[self.assetNames]

            self.var = VAR(data,self.assetNames)
            self.var.fit_ecm_coint(0)
            self.current_asset = 1
            if self.var.num_of_cointegration==0:
                self.var = None

    def update_current_asset(self,last_signal):
        if len(self.assetHistClose[self.assetNames[0]])>=2:
            for key in last_signal:
                self.current_asset += last_signal[key]*(self.assetHistClose[key][-1]-self.assetHistClose[key][-2])/self.capital
            print 'Current asset: %s'%self.current_asset

        
        
            

    def get_optimal_allocation(self,var_model,signal,T):

            
        num_of_assets = len(var_model.assetNames)
        X0 = np.zeros([num_of_assets,1])
        for i in range(num_of_assets):
            X0[i,0] = self.assetHistClose[var_model.assetNames[i]][-1]
        
        weight = var_model.solve(self.maturity-self.timeLeft,self.maturity,X0,1,1.1).flatten()
        #if self.current_asset<=1.01:
        #    weight = var_model.solve(T,X0,self.current_asset,1.01).flatten()
        #elif self.current_asset>1.01:
        #    self.timeLeft=-1
        #    return signal
            
            #weight = var_model.solve(T,X0,self.current_asset,self.current_asset+0.1).flatten()

        #weight = weight/np.abs(weight).sum()/len(self.var)
        

        for i in range(num_of_assets):
            signal[var_model.assetNames[i]] += weight[i]
        return signal


    def generate_trade_signal(self):
        signal = {}
        for key in self.assetNames:
            signal[key] = 0

        if self.var is None  or self.timeLeft<=1:
            self.fit_var()
            self.timeLeft = int(self.maturity)

        if self.var is not None:
            self.timeLeft -= 1

            signal = self.get_optimal_allocation(self.var,signal,self.timeLeft)


            for key in signal:
                signal[key] = signal[key]*self.capital


            


        return signal
        
            


def calculate_pnl(signal_df,data,assetNames):
    signal_df = signal_df[assetNames]
    data = data[assetNames]

    signal = signal_df.values[:-1,:]
    df = data.values
    df = np.diff(df,axis=0)
    pnl = df*signal
    pnl = np.cumsum(pnl,axis=0)
    combo_pnl = np.sum(pnl,axis=1)
    pnl = pd.DataFrame(pnl)
    pnl.columns = assetNames
    pnl.plot()
    plt.show()
    plt.plot(combo_pnl)
    plt.show()



    
    
    
    

    





def main():
    data = pd.read_csv('2800_2828.csv',index_col=0,parse_dates=True)
    data = np.log(data)
    signal_list = []

    assetNames = data.columns
    strategy = Strategy(assetNames)
    last_signal = {}
    for asset in assetNames:
        last_signal[asset] = 0

    for date in data.index:
        print "--------date: %s-------"%date
        for asset in assetNames:
            strategy.assetHistClose[asset].append(data.at[date,asset])


        strategy.update_current_asset(last_signal)
        
        signal = strategy.generate_trade_signal()
        #if strategy.timeLeft<strategy.maturity-1:
        #    signal = last_signal
        last_signal = signal
        print signal
        raw_input(3)
        signal_list.append(signal)

    signal_df = pd.DataFrame(signal_list)
    calculate_pnl(signal_df,data,assetNames)
        

        

    





if __name__ == '__main__':
    main()

