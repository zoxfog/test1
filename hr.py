#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import darts
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries


import da_transformations as da
import da_forecasting
from da_datasets import UnivariateDataset, get_dataset,get_m3,arange_m3



# In[8]:

for i in range(10):
    print("###################")
    print(i)
    print("###################")
    seed = np.random.randint(low = 0,high = 1000)
    print(seed)


    datasets = ['AirPassengers','AAPL','AusBeer','Electricity','Sunspots','HeartRate']
    datasets = ['HeartRate']
    data, seasonality = get_dataset(datasets[0]) 
    seasonality = 15

    models = ['RNN','N-BEATS-G','N-BEATS-I','TRANSFORMER','TCN']
    #models = ['RNN']
    split = 0.7
    reps = 1
    inlen = 12
    outlen = 6




    aug_n = 0 # number of operations
    aug_m = [] # magnitude per operation
    aug_p = [] # probability density function for the transformations
    baseline_result  = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True,
                seed = seed)      



    # In[ ]:


    aug_n = 1
    aug_m = [5]
    aug_p = [(np.array(list(da.ranges.keys()))=='Identity').astype(int)]
    iden_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True,
                seed=seed)



    # In[ ]:


    aug_n = 1
    aug_m = [1]
    aug_p = [(np.array(list(da.ranges.keys()))=='Jittering').astype(int)]
    jitter_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True,
                seed = seed) 



    # In[ ]:


    aug_n = 1
    aug_m = [4]
    aug_p = [(np.array(list(da.ranges.keys()))=='Noise Scale').astype(int)]
    ns_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True,
                seed = seed) 



    # In[ ]:


    aug_n = 1
    aug_m = [5]
    aug_p = [(np.array(list(da.ranges.keys()))=='Flip').astype(int)]
    flip_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True,
                seed=seed)



    # In[ ]:


    aug_n = 1
    aug_m = [7]
    aug_p = [(np.array(list(da.ranges.keys()))=='Trend Scale').astype(int)]
    trend_scale_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True,seed = seed)



    # In[ ]:


    aug_n = 1
    aug_m = [6]
    aug_p = [(np.array(list(da.ranges.keys()))=='Permutation').astype(int)]
    perm_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True, seed = seed)



    aug_n = 1
    aug_m = [7]
    aug_p = [(np.array(list(da.ranges.keys()))=='Scale').astype(int)]
    scale_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True, seed = seed)



    # In[ ]:


    aug_n = 1
    aug_m = [7]
    aug_p = [(np.array(list(da.ranges.keys()))=='Reverse').astype(int)]
    reverse_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True,seed = seed)



    # In[ ]:


    aug_n = 1
    aug_m = [4]
    aug_p = [(np.array(list(da.ranges.keys()))=='Smooth LS').astype(int)]
    ls_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True, seed = seed)



    # In[ ]:


    aug_n = 1
    aug_m = [4]
    aug_p = [(np.array(list(da.ranges.keys()))=='Smooth ETS').astype(int)]
    ets_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True, seed = seed)



    aug_n = 1
    aug_m = [7]
    aug_p = [(np.array(list(da.ranges.keys()))=='MBB').astype(int)]
    mbb_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True,seed = seed)



    aug_n = 1
    aug_m = [2]
    aug_p = [(np.array(list(da.ranges.keys()))=='Gaussian TW').astype(int)]
    gtw_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True, seed = seed)





    # In[ ]:


    aug_n = 1
    aug_m = [6]
    aug_p = [(np.array(list(da.ranges.keys()))=='DTS').astype(int)]
    dts_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True, seed = seed)



    aug_n = 1
    aug_m = [5]
    aug_p = [(np.array(list(da.ranges.keys()))=='DTS').astype(int)]
    ww_results = da_forecasting.run(models = models, dataset_names = datasets,
                split=split,
                inlen=inlen, outlen=outlen,
                reps=reps,
                aug_n_ops=aug_n , aug_mag=aug_m, aug_prob=aug_p,
                plot = True, seed = seed)
    
    






    results = []
    results.append(dts_results)
    results.append(gtw_results)
    results.append(mbb_results)
    results.append(ets_results)
    results.append(ns_results)
    results.append(ls_results)
    results.append(reverse_results)
    results.append(scale_results)
    results.append(perm_results)
    results.append(trend_scale_results)
    results.append(flip_results)
    results.append(jitter_results)
    results.append(iden_results)
    results.append(baseline_result)
    results.append(ww_results)

    aug_results = pd.concat(results)
    aug_results['seed']  = seed


    # In[ ]:


    aug_results.to_excel(datasets[0]+"/seed_"+str(seed)+"_aug_results2.xlsx")





