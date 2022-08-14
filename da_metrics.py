import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error 
from sklearn.metrics import mean_squared_error



def smape(A, F,axis=0):
    A = A.ravel()
    F = F.ravel()
    a = np.abs(A-F) 
    b = np.abs(A) + np.abs(F)
    if np.all(b)==False:
        raise Exception("Division with 0, ")
    return 200 * np.mean(a / b)


def mape(A, F):
    A = A.ravel()
    F = F.ravel()
    if np.all(A)==False:
        raise Exception("Division with 0, ")
    return 100 * np.mean(np.abs(A-F)  / A)

def rmse(A,F):
    A = A.ravel()
    F = F.ravel()
    return np.sqrt(mse(A, F))

def mse(A,F):
    return mean_squared_error(A, F)

#def mase(A,F,inlen,outlen,seasonality):
 #   a = np.sum(np.abs(A - F),axis = 1)
  #  b1_i = np.arange(seasonality,inlen + outlen,seasonality)
   # b2_i = np.arange(0,inlen + outlen - seasonality,seasonality)
    #b1 = A[:,b1_i]
    #b2 = F[b2_i]
    #b = (1/(inlen+outlen-seasonality))*np.sum(np.abs(b1-b2))
    #return (1/outlen)  * (a/b)
    
