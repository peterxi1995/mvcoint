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
        self.cointLookBack = 1000
        self.capital = 1e6
        self.maturity = 60
        self.timeLeft = 0
        self.assetHistClose = {}
        self.var = []
        for asset in self.assetNames:
            self.assetHistClose[asset] = deque([],maxlen=self.cointLookBack)

    def fit_var(self):
        self.var = []
        if len(self.assetHistClose[self.assetNames[0]]) == self.cointLookBack:
            print 'Fitting var....'
            data = pd.DataFrame(self.assetHistClose)
            data = data[self.assetNames]
            

            #for k in range(2,len(self.assetNames)+1):
            for k in range(2,3):
                for pair in combinations(self.assetNames,k):
                    temp_data = data[list(pair)]
                    var_model = VAR(temp_data,list(pair))
                    var_model.fit_ecm_coint(0)
                    if var_model.num_of_cointegration>0:
                        self.var.append(var_model)
                        #var_model.summary()

                    

    def get_optimal_allocation(self,var_model,signal):

            
        num_of_assets = len(var_model.assetNames)
        X0 = np.zeros([num_of_assets,1])
        for i in range(num_of_assets):
            X0[i,0] = self.assetHistClose[var_model.assetNames[i]][-1]
        
        weight = var_model.solve(self.maturity,X0,1,1.1).flatten()
        weight = weight/np.abs(weight).sum()/len(self.var)

        for i in range(num_of_assets):
            signal[var_model.assetNames[i]] += weight[i]
        return signal


    def generate_trade_signal(self):
        signal = {}
        for key in self.assetNames:
            signal[key] = 0

        if len(self.var)==0  or self.timeLeft<=0:
            self.fit_var()
            self.timeLeft = int(self.maturity)

        if len(self.var)>0:
            self.timeLeft -= 1

            for var_model in self.var: 
                signal = self.get_optimal_allocation(var_model,signal)


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
    data = pd.read_csv('HSI_banks.csv',index_col=0,parse_dates=True)
    data = np.log(data)
    signal_list = []

    assetNames = data.columns
    strategy = Strategy(assetNames)

    for date in data.index:
        print "--------date: %s-------"%date
        for asset in assetNames:
            strategy.assetHistClose[asset].append(data.at[date,asset])


        signal = strategy.generate_trade_signal()
        print signal
        signal_list.append(signal)

    signal_df = pd.DataFrame(signal_list)
    calculate_pnl(signal_df,data,assetNames)
        

        

    





if __name__ == '__main__':
    main()

