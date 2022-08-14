#!/usr/bin/env python
# coding: utf-8

# In[7]:


from logging.handlers import DatagramHandler
import sys
from os import getcwd
from os.path import basename, dirname
from tkinter.messagebox import YES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import random
import statsmodels
from statsmodels.tsa.seasonal import STL
from statsmodels.tools.eval_measures import rmse
import yfinance as yf
import sys


# blah lbsdfafasdfasfasfdasfasf

# In[2]:


import darts
from darts.models import NBEATSModel,BlockRNNModel,RNNModel,ExponentialSmoothing, TCNModel, TransformerModel,TFTModel
from statsmodels.tsa.seasonal import STL
from darts.utils.data import TrainingDataset, PastCovariatesTrainingDataset
from darts.utils.data import PastCovariatesInferenceDataset
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from darts.datasets import AirPassengersDataset, AusBeerDataset
from darts.datasets import SunspotsDataset, ElectricityDataset,HeartRateDataset
from darts.dataprocessing.transformers.boxcox import BoxCox

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

import logging
logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

import da_transformations as da




datasets = ['AirPassengers','AAPL','AusBeer','Electricity','Sunspots','HeartRate']
def get_dataset(name):
    if name == 'AirPassengers':
        series = AirPassengersDataset().load()
        series = TimeSeries.pd_series(series)
        seasonality = 12
    elif name == 'Electricity':
        series = ElectricityDataset(multivariate=False).load()
        #series = TimeSeries.pd_series(series)
        seasonality = 24
    elif name == 'AusBeer':
        series = AusBeerDataset().load()
        series = TimeSeries.pd_series(series)
        seasonality = 4
    elif name == 'Sunspots':
        series = SunspotsDataset().load()
        series = TimeSeries.pd_series(series)
        seasonality = 12
    elif name == 'AAPL':
        data = yf.download("AAPL", start="2017-11-01", end="2019-04-30")
        # avoid timestamps for compatibility
        series  = pd.Series(data["Close"].values)
        seasonality = 22
        #series  =TimeSeries.from_series(data,fill_missing_dates=True,freq = 'D')
        #series = fill_missing_values(series)
    elif name == 'HeartRate':
        series = HeartRateDataset().load()
        series = TimeSeries.pd_series(series)
        seasonality = 20
    else:
        print("TODO -  other datasets")
    return series, seasonality


datasets_m3 = ['M3Year','M3Month','M3Quart','M3Other']
def get_m3(name):
    if name=='M3Year':
        data = pd.read_excel("datasets/M3C.xls",sheet_name = 'M3Year')
    if name=='M3Quart':
        data =  pd.read_excel("datasets/M3C.xls",sheet_name = 'M3Quart')
    if name=='M3Month':
        data = pd.read_excel("datasets/M3C.xls",sheet_name = 'M3Month')
    if name=='M3Other':
        data = pd.read_excel("datasets/M3C.xls",sheet_name = 'M3Other')        
    return data




def arange_testset(data, inlen, outlen, split_index, inds = None):
    #print(split_index)

    val_input_index  = split_index-(inlen+outlen-1)   
    #print(val_input_index)
    seq = []
    test_size = len(data[split_index:])
    for i in range(test_size):
        series_i = i + val_input_index
        series_vals = data[series_i:series_i+inlen]
        
        if inds is not None:
            series_inds = pd.DatetimeIndex(inds[series_i:series_i+inlen])
            #print(series_vals)
            #print(series_inds)
            ts = TimeSeries.from_times_and_values(series_inds, series_vals)

        else:
            ts = TimeSeries.from_values(series_vals)

        ts = TimeSeries.from_times_and_values(series_inds, series_vals)
        seq.append(ts)
    return seq


def arange_m3(data,inlen = None, h_multi = 2,shift_time=False, use_scaler = True):
    start_i = 6
    forecast_h = data.iloc[0]['NF']
    

    # if monthly
    if forecast_h == 18:
        seasonality = 12
    # yearly, quarterly and other
    else:
        seasonality = 4
    
    if inlen==None:
        inlen = forecast_h*h_multi

    # get the scaler, fit it to the data
    if use_scaler == True:
        m3scaler = MinMaxScaler()
        data_scaler = data.iloc[:,start_i:].to_numpy().ravel()
        data_scaler = data_scaler[~np.isnan(data_scaler)]
        m3scaler.fit(data_scaler.reshape(-1,1))
    else:
        m3scaler = None

    # for each timeseries present in M3
    for i in range(len(data)):
        seq_length =  data.iloc[i]['N']
        seq = data.iloc[i][start_i:start_i + seq_length ].values
        train_seq = seq[:-forecast_h].astype(float)      
        test_seq = seq[-forecast_h:].astype(float)  

       # test_ts = np.expand_dims(test_ts,1)
        #train_ts = np.expand_dims(train_ts,1)
        if use_scaler == True:
            train_ts = m3scaler.transform(train_seq.reshape(-1,1)).ravel()
        else:
            train_ts = train_seq

            

        if i == 0:
            train_dataset = UnivariateDataset(inlen,forecast_h,train_seq,shift_time=shift_time, scaler = m3scaler)
            trainset_ts = [TimeSeries.from_values(train_ts)]
            trainset = [train_seq]
            testset = np.expand_dims(test_seq,1)
        else:
            train_dataset.create_dataset(train_seq,inlen,forecast_h,reset = False)
            trainset_ts.append(TimeSeries.from_values(train_ts))
            trainset.append(train_seq)
            testset = np.concatenate([testset,np.expand_dims(test_seq,1)],axis =1)
    
    return train_dataset,trainset,trainset_ts,testset.T
   





class UnivariateDataset(PastCovariatesTrainingDataset):
    def __init__(self, inlen, outlen, data, seasonality = 12, shift_time = False, scaler = None):
        super().__init__()
        
        
        self.inlen = inlen
        self.outlen = outlen
        self.seasonality = seasonality

        self.shift_time = shift_time
        self.data = data.ravel()

        self.scaler = scaler

        self.dataset = self.create_dataset(data, inlen, outlen)

        
        # length of the origiginal dataset
        self.orig_len = len(self.dataset)
        
    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    
    # returns the length of the original dataset without augmentations
    def original_len(self):
        return self.orig_len
    
    def create_dataset(self, data, inlen, outlen, reset = True):
        if reset==True:
            dataset = []
        else:
            dataset = self.dataset

        for i in range(len(data)-(inlen+outlen)+1):
            if self.shift_time == True:
                x = np.expand_dims(data[i:i+inlen],1)
                y = np.expand_dims(data[i+(inlen-outlen):i+(2*inlen-outlen)],1)
            else:
                x = np.expand_dims(data[i:i+inlen],1)
                y = np.expand_dims(data[i+inlen:i+inlen+outlen],1)

            if self.scaler is not None:
                x = self.scaler.transform(x)
                y = self.scaler.transform(y)
                                        
            dataset.append((x,None,None,y))
        return dataset

    

    
    
    # pooling augmentation strategy - augments the whole time-series length
    def augment(self,  magnitude_ops: list, p_ops = [] ,n_ops = 1, inplace = True, size = 1, plot = True):
        
        if len(p_ops)!=n_ops:
            raise Exception("The number of probability distributions do not match the number of given operations")
        if len(magnitude_ops)!=n_ops:
            raise Exception("The number of mag operations do not match the number of given operations")

                        
        # remove any old augmentations 
        self.dataset = self.dataset[:self.original_len()]        
        op_trans_names = []
        augmented_data = self.data
        
        # for each operation
        for i_op in range(n_ops):
            
            mag = magnitude_ops[i_op]            
            if mag>da.mag_bins:
                raise Exception("mag argument is larger than the max magnitude bin")
                
            op_prob = p_ops[i_op]            
            if len(op_prob)!= len(da.ranges):
                raise Exception("The number of probablities in p must match the number of ranges")
            if np.round(np.sum(p_ops[i_op]),3)!=1:
                raise Exception("The CDF dos not sum to 1")
                
            
            # for each transformation with a corresponding positive probability
            for i_prob in range(len(op_prob)):   
                # skip if the probability is 0
                if op_prob[i_prob]==0:
                    continue
                prob = op_prob[i_prob]
                trans_name = list(da.ranges.keys())[i_prob]
                op_trans_names.append(trans_name)
                    
                # get a sample size for this transformation
                sample_size = int(self.original_len()*prob*size) 
                
                    
                augmented_data = da.apply_op(augmented_data,
                                            trans_name,
                                            mag,
                                            self.seasonality) 
                
                # todo
                
           #     augmented_dataset = self.__create_dataset(augmented_data,
            #                                   self.inlen,
             #                                  self.outlen) 
                
                # sample the augmented data for this transformation
                
               # sampled_indicies=np.random.choice(len(augmented_dataset),sample_size,replace = True)
                #sampled_augment = [augmented_dataset[i] for i in sampled_indicies]
                
                
        augmented_dataset = self.create_dataset(augmented_data,
                                                  self.inlen,
                                                  self.outlen) 
        # append it to the train dataset (note: + means append for python lists)
        sampled_augment = augmented_dataset
        self.dataset = self.dataset + sampled_augment
                
        if plot==True:
            plt.plot(self.data)
            plt.plot(augmented_data)
            plt.title("Final Augmentation,  magnitudes {} ".format(magnitude_ops))
            plt.xlabel('timestep')
            plt.show()
