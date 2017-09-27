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
        # How long data to use to fit a model
        self.cointLookBack = 250
        # Maximum deployable capital
        self.capital = 1
        # The capitial that's willing to be risked at entry
        self.entry_capital = 0.25
        self.target_return = 1.001

        # Set investment horizon
        self.maturity = 5
        # Set time index
        self.time_index = 0
        # Record price
        self.assetHistClose = {}
        self.var = None
        self.current_asset = 1
        for asset in self.assetNames:
            self.assetHistClose[asset] = deque([],maxlen=self.cointLookBack)

    def fit_var(self):
        # get a var model with ecmcoint, see if it is appropriate to make an entry now
        self.var = None
        self.time_index = 0

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
            else:
                X0 = np.matrix([[self.assetHistClose[self.assetNames[0]][-1]],[self.assetHistClose[self.assetNames[1]][-1]]])
                self.var.ecmcoint.get_coefficients(self.maturity,X0,self.target_return)
                weight = self.var.ecmcoint.solve_time_consistent(0,self.maturity,X0,1,self.target_return)
                print 'Suggested entry weight: %s'%weight
                if np.abs(weight).sum()<=self.entry_capital:
                    print "Right timing, risk capital"
                else:
                    print "Currently not a good time to open position"
                    self.var = None
                
    def get_optimal_allocation(self):
        if self.time_index>= self.maturity:
            print "Came to end, clear and out"
            weight = np.zeros([len(self.assetNames),1])
            self.var = None
            
        elif self.var is not None:
            # var model is available, then start to act
            
            X_t = np.matrix([[self.assetHistClose[self.assetNames[0]][-1]],[self.assetHistClose[self.assetNames[1]][-1]]])
            weight =  self.var.ecmcoint.solve_time_consistent(self.time_index,self.maturity,X_t,self.capital,self.target_return)
            print 'Weight suggested: %s'%weight
            if self.current_asset>=self.target_return:
                print "Objective achieved, clear and out"
                weight = np.zeros([len(self.assetNames),1])
                self.var = None
            elif np.abs(weight).sum()>1:
                print "Requesting more capital than endured, clear and out"
                weight = np.zeros([len(self.assetNames),1])
                self.var = None
            

            self.time_index +=1
        else:
            weight = np.zeros([len(self.assetNames),1])


        result = {}
        for i in range(len(self.assetNames)):
            result[self.assetNames[i]] = weight[i,0]


        return result


    def update_current_asset(self,last_signal):
        if len(self.assetHistClose[self.assetNames[0]])>=2:
            for key in last_signal:
                self.current_asset += last_signal[key]*(self.assetHistClose[key][-1]-self.assetHistClose[key][-2])/self.capital
            print 'Current asset: %s'%self.current_asset
        
            


    def generate_trade_signal(self):
        signal = {}
        for key in self.assetNames:
            signal[key] = 0

        if self.var is None:
            self.fit_var()

        if self.var is not None:
            # Cointegration premium is found

            signal = self.get_optimal_allocation()

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
    print pnl
    
    pnl.plot()
    plt.show()
    plt.plot(combo_pnl)
    plt.show()



    
    
    
    

    





def main():
    data = pd.read_csv('SPY_IVV.csv',index_col=0,parse_dates=True)
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
        signal_list.append(signal)

    signal_df = pd.DataFrame(signal_list)
    calculate_pnl(signal_df,data,assetNames)
        

        

    





if __name__ == '__main__':
    main()

