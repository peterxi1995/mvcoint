from __future__ import division
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.linalg import expm
import scipy
from scipy.integrate import quad
from johansen import Johansen
import statsmodels.api as sm
import matplotlib.pyplot as plt


'''
Code for time consistent and precommitment policy of mean variance optimal portfolio under cointegration

The solution is an optimal control policy at time t

Reference: Chiu & Wong (2011), Chiu & Wong (2015)
<Mean-variance portfolio selection of cointegrated asset>
<Dynamic cointegrated pairs trading: Mean-variance time-consistent strategies>


Author: Peter Xi

'''


class VAR:
    '''
    Use Johansen's test and OLS to fit a VAR(1) model
    '''

    def __init__(self,data,assetNames):
        # Only use python implementation
        # data should be a pandas dataframe
        NUM_OF_ASSETS = data.shape[1]
        self.assetNames = assetNames
        johansen_model = Johansen(x=data,model=2,k=1,trace=True,significance_level=0)
        eigenvectors, rejected_r_values = johansen_model.johansen()
        self.eigenvectors = eigenvectors
        self.data = data

        num_of_cointegration = len(rejected_r_values)
        self.num_of_cointegration = num_of_cointegration

        if num_of_cointegration==0:
            # Suppose there is no cointegration relationship, reduce the model to general log-normal model with low statistical power and premium
            A = np.zeros([NUM_OF_ASSETS,NUM_OF_ASSETS])
            theta = np.zeros([NUM_OF_ASSETS,1])
            residuals = []
            for i in range(NUM_OF_ASSETS):
                Y = np.diff(data.values[:,i])
                theta[i,0] = np.mean(Y)
                residuals.append(Y-theta[i,0])

            cov = np.cov(residuals,ddof=1)
            self.theta,self.A,self.cov = theta,-A,cov

            
        else:
            length = len(data)-1
            coint_series = np.zeros([length,num_of_cointegration])
            for i in range(num_of_cointegration):
                coint_series[:,i] =  (data.values[:length].dot(eigenvectors[:,i])).flatten()

            X = sm.add_constant(coint_series)
            
            # Then run regression on coint_series
            

            theta = np.zeros([NUM_OF_ASSETS,1])
            A = np.zeros([NUM_OF_ASSETS,NUM_OF_ASSETS])
            residuals = []
            for i in range(NUM_OF_ASSETS):
                # diff_X_i
                Y = np.diff(data.values[:,i])
                model = sm.OLS(Y,X)
                results = model.fit()
                theta[i,0] = results.params[0]
                resid = Y-model.predict(results.params ,X)
                residuals.append(resid.flatten())

                for k in range(NUM_OF_ASSETS):
                    # coefficient for asset k 
                    for n in range(num_of_cointegration):
                        # Consider the nth cointegration relationship
                        A[i,k] += results.params[n+1]*eigenvectors[k,n]
            cov = np.cov(residuals,ddof=1)
            self.theta,self.A,self.cov = theta,-A,cov


        self.sigma = np.linalg.cholesky(self.cov)
        #self.A = self.A*np.sqrt(252)
        #self.sigma = self.sigma*np.sqrt(252)
        #self.theta = self.theta*np.sqrt(252)
        #self.cov = self.cov*252

    def summary(self):
        print '--------------------------------------------------------------------------------'
        print 'Num of cointegration: %s'%(self.num_of_cointegration)
        print 'Loading matrix: \n%s'%(self.eigenvectors)
        print '\nmodel: Ln(X) = (theta - ALn(X))dt + sigmadWt\n'
        print 'theta:\n %s'%self.theta
        print 'A: \n%s'%self.A
        print "sigma: \n%s"%self.sigma
        print "covariance: \n%s"%self.cov
        print '--------------------------------------------------------------------------------'

    def plot(self):
        for i in range(self.num_of_cointegration):
            combination = self.data*self.eigenvectors[:,i]
            plt.plot(combination.sum(axis=1))
        if self.num_of_cointegration>0:
            plt.show()












    def fit_ecm_coint(self,r):
        self.ecmcoint = ECMCoint(self.theta,self.sigma,self.A,r)
    def solve(self,t,T,X_t,Y_0,Y_hat):
        return self.ecmcoint.solve_time_consistent(t,T,X_t,Y_0,Y_hat)


class ECMCoint:
    '''
    Mean variance portfolio optimizer under the existance of cointegartion

    The solver could also work if there is no cointegration relationship detected. In which case the matrix A should be reduced to a zero matrix.

    The pre-commitment solver is highly unstable and would diverge in finite time. It is recommended to use the time-consistent solver which is much more stable though yield lower sharpe ratio
    
    '''


    def __init__(self,theta,sigma,A,r):
        self.theta = theta
        self.sigma = sigma
        self.A = A
        self.r = r
        self.K_0,self.N_0,self.M_0 = None,None,None


    def solve_pre_commit(self,T,X0,Y0,R):
        # give out a solution that is not time-consistent

        NUM_OF_ASSETS = len(self.A)
        temp_1 = np.concatenate((-self.A,-self.A.dot(self.sigma).dot(self.sigma.T).dot(self.A.T)),axis=1)
        temp_2 = np.concatenate((2*inv(self.sigma.dot(self.sigma.T)),self.A.T),axis=1)
        temp_3 = np.concatenate((np.eye(NUM_OF_ASSETS),np.zeros([NUM_OF_ASSETS,NUM_OF_ASSETS])),axis=0)
        M_mat = np.concatenate((temp_1,temp_2),axis=0)
        COV = self.sigma*self.sigma.T
        D = np.diag(np.diag(COV))


        THETA = (0.5*self.A.dot(D)-self.A*self.r).dot(np.ones([NUM_OF_ASSETS,1]))

        
        temp_H = self.A.dot(COV).dot(self.A.T)

        def K(t):
            M_coef = expm(M_mat*(T-t))
            temp = M_coef.dot(temp_3)
            R_tau = temp[:NUM_OF_ASSETS,:]
            Z_tau = temp[NUM_OF_ASSETS:,:]
            K_t = Z_tau.dot(inv(R_tau))
            return K_t

        def H(s):
            return -(self.A+temp_H.dot(K(s)))

        q,P = np.linalg.eig(H(0))
        q = np.diag(q)
        P_inv = inv(P)



        def PhiH(s):
            elements = []
            for i in range(NUM_OF_ASSETS):
                def G(s):
                    return np.diag(P_inv.dot(H(s)).dot(P))[i]
                elements.append(quad(G,0,s)[0])
            temp = -np.diag(elements)
            temp = expm(temp)
            return P.dot(temp).dot(P_inv)

        def N(s):
            elements = []
            for i in range(NUM_OF_ASSETS):
                def grand(s):
                    return THETA.T.dot(K(s).T).dot(PhiH(s))[0,i]
                elements.append(quad(grand,0,T-s)[0])
            temp = np.matrix(elements)
            return temp.dot(inv(PhiH(T-s))).T

        temp_mat = self.A.dot(COV).dot(self.A.T)

        def M(s):
            def grand(s):
                N_s = N(s)
                return N_s.T.dot(THETA)+0.5*np.trace(temp_mat.dot(K(s)))+0.5*N_s.T.dot(temp_mat).dot(N_s)
            return quad(grand,0,T-s)[0]
       

        ##########Derive the optimal control#######################
        b_0 = self.theta - self.A.dot(X0) + 0.5*D.dot(np.ones([2,1]))
        beta_0 = b_0 - self.r*np.ones([NUM_OF_ASSETS,1])

        K_0 = self.K_0
        if K_0 is None:
            K_0 = K(0)
            print 'K_0: \n%s'%K_0
        N_0 = self.N_0
        if N_0 is None:
            N_0 = N(0)
            print 'N_0: \n%s'%N_0
        M_0 = self.M_0
        if M_0 is None:
            M_0 = M(0)
            print 'M_0: \n%s'%M_0


        p_0 = float(np.exp(2*self.r*T + \
                 -0.5*beta_0.T.dot(K_0).dot(beta_0)-N_0.T.dot(beta_0)-M_0))
        eta_0 = -(2*K_0.dot(beta_0)+N_0).T.dot(self.A).dot(self.sigma)*(p_0)
        lambda_star = (2*p_0*np.exp(-self.r*T)*(Y0-R*np.exp(-self.r*T))) / \
                (p_0*np.exp(-2*self.r*T)-1)

        u = -inv(COV).dot( beta_0 + self.sigma.dot(eta_0.T)/p_0 ).dot(Y0-(R+lambda_star/2)*np.exp(-self.r*T))

        arbitrage_index = float(0.5*beta_0.T.dot(K_0).dot(beta_0)+N_0.T.dot(beta_0)+M_0)
        self.arbitrage_index = arbitrage_index
        #price_of_risk = np.sqrt(1/(p_0*np.exp(-2*self.r*T))-1)
        price_of_risk = np.sqrt(np.exp(arbitrage_index)-1)
        self.price_of_risk = price_of_risk
        self.K_0 = K_0
        self.N_0 = N_0
        self.M_0 = M_0
        print ("Suggested Allocation : %s"%u.flatten())
        print ("Implied Sharpe: %s"%self.price_of_risk)
        print ("Arbitrage Index: %s"%self.arbitrage_index)

    def get_coefficients(self,maturity,X0,Y_hat):
        # Get the coefficients needed to calculate the optimal allocation
        # This should be called at time 0
        cov = self.sigma.dot(self.sigma.T)
        cov_inv = inv(cov)
        NUM_OF_ASSETS = len(self.A)

        D = np.diag(np.diag(cov))

        THETA = 0.5*self.A.dot(D).dot(np.ones([NUM_OF_ASSETS,1])) - self.r*self.A.dot(np.ones([NUM_OF_ASSETS,1]))
        alpha = self.theta - self.A.dot(X0) + 0.5*D.dot(np.ones([NUM_OF_ASSETS,1])) - self.r*np.ones([NUM_OF_ASSETS,1])

        # aim is to get the lambda
        def K_hat(t,T):
            return 2*cov_inv*(T-t)
        def N_hat(t,T):
            return (THETA.T.dot(cov_inv)*(T-t)**2).T
        def M_hat(t,T):
            return (1/3)*float(THETA.T.dot(cov_inv).dot(THETA))*(T-t)**3 + np.trace(self.sigma.T.dot(self.A.T).dot(cov_inv).dot(self.A).dot(self.sigma))/2 *(T-t)**2
        def H_hat(T):
            return 0.5*alpha.T.dot(K_hat(0,maturity)).dot(alpha) + N_hat(0,maturity).T.dot(alpha)+M_hat(0,maturity)

        H_hat_T = float(H_hat(maturity))

        self.lambda = (Y_hat - np.exp(self.r*maturity)*1)/H_hat_T


    def solve_time_consistent(self,t,maturity,X_t,Y_0,Y_hat):
        '''
        give out a solution that is time-consistent
        solution follows the scheme given by Chiu(2015)
        
        The optimal control u only concerns the current price level, time left, initial wealth and target;
        it does not care about the current state of wealth

        Must be done after self.get_coefficients have been called
        '''
        cov = self.sigma.dot(self.sigma.T)
        cov_inv = inv(cov)
        NUM_OF_ASSETS = len(self.A)

        D = np.diag(np.diag(cov))

        THETA = 0.5*self.A.dot(D).dot(np.ones([NUM_OF_ASSETS,1])) - self.r*self.A.dot(np.ones([NUM_OF_ASSETS,1]))

        # get optimal allocation

        def K_hat(t,T):
            return 2*cov_inv*(T-t)
        def N_hat(t,T):
            return (THETA.T.dot(cov_inv)*(T-t)**2).T

        alpha = self.theta - self.A.dot(X_t) + 0.5*D.dot(np.ones([NUM_OF_ASSETS,1])) - self.r*np.ones([NUM_OF_ASSETS,1])

        # H_hat is only needed for (H_hat(0,alpha(0),T)):

        # The Y_0 should not have been changed
        u = self.lambda * np.exp(self.r*(maturity-t)) * ((cov_inv+self.A.T.dot(K_hat(t,maturity))).dot(alpha)+self.A.T.dot(N_hat(t,maturity)))

        return u



        

if __name__ == '__main__':
    '''    
    theta = np.matrix([[0.1],[0.2]])
    sigma = np.diag([0.2,0.2])
    A = 1/2 * np.ones([2,2])
    print A.dot(A)
    model = ECMCoint(theta,sigma,A,0.03)
    model.solve_time_consistent(0.5,np.matrix([[np.log(1)],[np.log(2)]]),1,1.1)
    '''    
    data = pd.read_csv("SPY_IVV.csv",index_col=0)
    data = np.log(data)
    model = VAR(data,data.columns)
    model.summary()
    model.fit_ecm_coint(0)
    print model.solve(100,np.matrix([[data.values[-1,0]],[data.values[-1,1]]]),1,1.2)
    

