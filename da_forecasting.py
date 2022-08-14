#!/usr/bin/env python
# coding: utf-8

# In[7]:


from re import I
import sys
from os import getcwd
from os.path import basename, dirname
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from scipy import stats
import random
import statsmodels
from statsmodels.tsa.seasonal import STL

import yfinance as yf
import argparse
import sys





# In[2]:


import darts
from darts.models import NBEATSModel,BlockRNNModel,RNNModel,ExponentialSmoothing, TCNModel, TransformerModel,TFTModel
from statsmodels.tsa.seasonal import STL
from darts.utils.data import TrainingDataset, PastCovariatesTrainingDataset
from darts.utils.data import PastCovariatesInferenceDataset
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from darts.datasets import AirPassengersDataset, AusBeerDataset
from darts.dataprocessing.transformers.boxcox import BoxCox

from sklearn.preprocessing import MinMaxScaler


import logging
logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

import da_transformations as da
from da_datasets import UnivariateDataset, get_dataset, arange_testset,get_m3,arange_m3
from da_metrics import mape,smape,rmse,mse
from da_models import get_modelM3,get_model



# DO NOT RUN main(), not complete
def main(args):

    parser = argparse.ArgumentParser(description="Train Models n times")
    parser.add_argument("-m", "--model",type=list, required=True)
    parser.add_argument("-d", "--dataset_name", type=list, required=True)
    parser.add_argument("-s", "--split",type=float, default = 0.7, required=False)
    parser.add_argument("-i", "--inlen",type=int, default = 12, required=False)
    parser.add_argument("-o", "--outlen",type=int, default = 6, required=False)
    parser.add_argument("-r", "--reps",type=int, default = 1, required=False)
    args = parser.parse_args(args)
   
    aug_n_ops = 0
    aug_mag = [5]
    aug_prob = []    
    plot = True    
    
    run(models = [args.model],
        dataset_names = [args.dataset_name],
        split = args.split,
        inlen = args.inlen,
        outlen = args.outlen,
        reps = args.reps,
        aug_n_ops = aug_n_ops,
        aug_mag = aug_mag,
        aug_prob = aug_prob,
        plot = plot)
    
    

def run(models: list, dataset_names: list,
        split=0.7,
        inlen=12, outlen=6,
        reps=1,
        aug_n_ops=0, aug_mag=[5], aug_prob=[],
        plot = True,
        seed = 23):

    results = pd.DataFrame(columns = ['Dataset',
                                    'Model',
                                    'avg MAPE',
                                    'avg RMSE',
                                    'avg sMAPE',
                                    'std MAPE',
                                    'std RMSE',
                                    'std sMAPE',
                                    'n ops',
                                    'mag',
                                    'op prob'])


    for dataset in dataset_names: 
                
        data, seasonality = get_dataset(dataset)
        split_index = int(split*len(data))
        train = data[:split_index]
        val = data[split_index:]
        
        scaler = MinMaxScaler() 
        scaler = scaler.fit(train.values.reshape(-1,1))
        #scaled_train = scaler.transform(train.values.reshape(-1,1)).ravel()
        #scaled_val = scaler.transform(val.values.reshape(-1,1)).ravel()
        #scaled_data = scaler.transform(data.values.reshape(-1,1)).ravel()


        #val_set = arange_testset(scaled_data, inlen, outlen,split_index, inds = data.index)
                

        for model_name in models:
            
            # tcn requires a different organization of the data in UnivariateDataset 
            is_tcn = True if model_name=='TCN' else False
                                    
            #TS_data = TimeSeries.from_times_and_values(data.index,scaled_data)
            train_dataset = UnivariateDataset(inlen = inlen,
                                              outlen = outlen,
                                              data = train,
                                             seasonality = seasonality,
                                             shift_time = is_tcn,
                                             scaler = scaler )

            scaled_train = scaler.transform(train.values.reshape(-1,1)).ravel()
            scaled_val = scaler.transform(val.values.reshape(-1,1)).ravel()
            scaled_data = scaler.transform(data.values.reshape(-1,1)).ravel()
            val_set = arange_testset(scaled_data, inlen, outlen,split_index, inds = data.index)

            # if number of operations is greater than 0
            if aug_n_ops>0:
                train_dataset.augment(aug_mag,
                                      n_ops = aug_n_ops,
                                      p_ops = aug_prob,
                                      size = 1,
                                      plot = plot)
            list_mape = []
            list_rmse = []
            list_smape = []
            for rep in range(reps):
                
                # initialize a new model with new params
                model = get_model(model_name, inlen, outlen, seed)
                model.fit_from_dataset(train_dataset)
                
                # val first input (inlen) index
                val_input_index  = split_index-(inlen+outlen-1)
                
              #  print("TS_data[val_input_index:]")
               # print(TS_data[val_input_index:][inlen+1])

                #print("inlen+1")
                #print(inlen+1)
                
                # val first value in output/horizon length (outlen), 
                # note: only the last horizon value in outlen is addressed


                fcast_scaled = model.predict(n = outlen,
                                              series = val_set,
                                              verbose=False)


                # take the last of outlen predicted values, and create a series 
                fcast =[]                             
                for ser in fcast_scaled:
                    ser_unscaled = scaler.inverse_transform(ser.values())
                    fcast.append(ser_unscaled[-1])
                

                fcast = pd.Series(data = fcast, index = val.index)

  
            #    fcast_scaled = model.historical_forecasts(
             #                               TS_data[val_input_index:],
              #                              start=inlen,
               #                             forecast_horizon=outlen,
                #                            verbose=False,
                 #                           retrain=False)


                #fcast = scaler.inverse_transform(fcast_scaled.values())
                fcast = pd.Series(data = fcast.ravel(), index = val.index)
                
                if plot==True:
                    plt.plot(fcast)
                                
                mape_ = np.round(mape(val, fcast),5)
                rmse_ = np.round(rmse(val, fcast),3)
                smape_ = np.round(smape(val, fcast),3)

                list_mape.append(mape_)
                list_rmse.append(rmse_)
                list_smape.append(smape_)

            row = pd.DataFrame(data = {'Dataset': [dataset],
                                       'Model': [model_name],
                                      'avg MAPE': [np.mean(list_mape)],
                                      'avg RMSE': [np.mean(list_rmse)],
                                      'avg sMAPE': [np.mean(list_smape)],
                                      'std MAPE': [np.std(list_mape)],
                                      'std RMSE': [np.std(list_rmse)],
                                      'std sMAPE': [np.std(list_smape)],
                                      'n ops':[aug_n_ops],
                                      'mag':[aug_mag],
                                      'op prob':[str(list(aug_prob))]})
            
            results = pd.concat([results,row])

            if plot==True:
                plt.plot(data)
                plt.title(model_name)               
                plt.show()                                                
    return results 




    
    

def runM3(models: list, dataset_names: list,
        h_multi = 2,
        reps=1,
        aug_n_ops=0, aug_mag=[5], aug_prob=[],
        plot = True,
        seed = 23,
        use_scaler = True):

    results = pd.DataFrame(columns = ['Dataset',
                                    'Model',
                                    'avg MAPE',
                                    'avg RMSE',
                                    'avg sMAPE',
                                    'std MAPE',
                                    'std RMSE',
                                    'std sMAPE',
                                    'n ops',
                                    'mag',
                                    'op prob'])


    for dataset in dataset_names: 

                
        data = get_m3(dataset)
        
        s = [77,68,55]
        plot_i = np.random.choice(len(data),size = s)
                       
        for model_name in models:
            
            # tcn requires a different organization of the data in UnivariateDataset 
            is_tcn = True if model_name=='TCN' else False

            # the data in train_dataset and valx_ts is scaled
            train_dataset,valx,valx_ts,valy = arange_m3(data,
                                                inlen = None,
                                                h_multi = h_multi,
                                                shift_time = is_tcn,
                                                use_scaler = use_scaler)
            if use_scaler == True:                                    
                scaler = train_dataset.scaler
                                    
           # data_for_scaling = np.array(train_dataset.dataset,dtype=object)
            # concatenate only the train examples
            #data_for_scaling = np.concatenate(a[:,0])
            #scaler = MinMaxScaler()
            #np.min(np.concatenate(data_for_scaling[:,0]))
            # if number of operations is greater than 0
           # if aug_n_ops>0:
            #    train_dataset.augment(aug_mag,
             #                         n_ops = aug_n_ops,
              #                        p_ops = aug_prob,
               #                       size = 1,
                #                      plot = plot)
            list_mape = []
            list_rmse = []
            list_smape = []
            for rep in range(reps):
                
                # initialize a new model with new params
                outlen = len(valy[0]) 
                inlen = outlen*h_multi  

                model = get_modelM3(model_name, inlen, outlen, seed)
                model.fit_from_dataset(train_dataset)
                
 


                fcast_ts = model.predict(n = outlen,
                                    series = valx_ts,
                                    verbose=False)
                #fcast_output = fcast.values()
                #print("fcast_ts")
                #print(len(fcast_ts))
                


                # take the last of outlen predicted values, and create a series 
                fcast = []  


                #for i in s:
                 #   fcast =  fcast_ts[i].values()  
                  #  plt.plot(valy[i])  
                   # plt.plot(fcast)  
                    #plt.show()

               # print("3333")
               # print(fcast.shape)  
                fcast =   fcast_ts[0].values()                
                for ser in fcast_ts[1:]:
                    fcast = np.concatenate([fcast,ser.values() ],axis=1)
                #    ser_unscaled = scaler.inverse_transform(ser.values())
                  #  fcast.append(ser_unscaled[-1])

                fcast = fcast.T
                if use_scaler==True:
                    fcast = scaler.inverse_transform(fcast)
               # print("fcast")
               # print(fcast.shape)
               # fcast = pd.Series(data = fcast, index = val.index)
                for i in s: 
                    plt.plot(valy[i])  
                    plt.plot(fcast[i])  
                    plt.show()

                plt.plot(valy[s[-1]])  
                plt.plot(fcast[s[-1]])  
                plt.show()

                #fcast = pd.Series(data = fcast.ravel(), index = val.index)

                if plot==True:
                    #temp = np.concatenate([valx[plot_i],fcast[plot_i]],axis =1)
                    for i in s:
                        #temp_ser = pd.Series(data = temp[i], index = range(len(temp[i])))
                        len_x = len(valx[i])
                        x = np.arange(len_x,len_x+outlen)
                        y = fcast[i]
                        #print(x)
                        #print(y)
                        plt.plot(x,y)

                #print("val")
                #print(valy)
                #print("fcast")
               # print(fcast)
                mape_ = np.round(mape(valy, fcast),5)
                rmse_ = np.round(rmse(valy, fcast),3)
                smape_ = np.round(smape(valy, fcast),3)
                #mase_ = np.round(mase(valy, fcast, inlen,outlen),3)

                list_mape.append(mape_)
                list_rmse.append(rmse_)
                list_smape.append(smape_)

            row = pd.DataFrame(data = {'Dataset': [dataset],
                                       'Model': [model_name],
                                      'avg MAPE': [np.mean(list_mape)],
                                      'avg RMSE': [np.mean(list_rmse)],
                                      'avg sMAPE': [np.mean(list_smape)],
                                      'std MAPE': [np.std(list_mape)],
                                      'std RMSE': [np.std(list_rmse)],
                                      'std sMAPE': [np.std(list_smape)],
                                      'n ops':[aug_n_ops],
                                      'mag':[aug_mag],
                                      'op prob':[str(list(aug_prob))]})
            
            results = pd.concat([results,row])

            if plot==True:
            
                for i in s:
                    temp = np.concatenate([valx[i].ravel(),valy[i].ravel()])
                    plt.plot(temp)
                plt.title(model_name)               
                plt.show()
                
                                                      
    return results 
        


