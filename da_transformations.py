# %% [markdown]
# 
# ## TODO: 
# * FFT based augmentations
# * DTW Barycenter Averaging
# * SMOTE for time-series
# * CZ

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL
import statsmodels
import random
from scipy import stats
from scipy import  special
from sklearn.preprocessing import MinMaxScaler



# %%

# should be even
mag_bins  = 8
ranges = {"Identity":None ,
          "Jittering":np.round(np.linspace(0.01, 0.1 , num=mag_bins ),3),
          "Trend Scale": np.round(np.linspace(0, 2, num=mag_bins  ),2),
          "Noise Scale":np.round(np.linspace(0.1, 1.5, num=8 ),2),
          "Flip": None,
          "Permutation": np.arange(2+mag_bins,2,-1),
          "Scale": np.round(np.linspace(0.5, 1.5, num=mag_bins ),2),
          "Reverse": None,
          "Smooth LS": np.round(np.geomspace(0.1, 10, num=mag_bins),2),
          "Smooth ETS": np.round(np.geomspace(1, 0.2, num=mag_bins),2),
          "MBB": np.round(np.linspace(0, 1, num=mag_bins),2),
          "Gaussian TW":np.round(np.linspace(0.2, 1, num=mag_bins),3),
          "DTS":np.arange(1,mag_bins+1),
          "Window Warping": np.concatenate([np.linspace(0.5,0.95,int(mag_bins/2)),
                                            np.linspace(1.1,2,mag_bins - int(mag_bins/2))])       
         # "DTS":np.arange(8,0,-1) #unnormalized
          }

'''
Injects noise in to the time series, the noise is sampled from N(0,sigma^2).
data: shape (N,) univariate time series
sigma: the sigma of the normal distribution, suggested range  [0,mean (t-1) difference]

Reference: Data Augmentation of Wearable Sensor Data for Parkinson’s
Disease Monitoring using Convolutional Neural Networks
''' 
def jittering(data, sigma):  
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    mu = 0
    noise = np.random.normal(mu, sigma, len(data))
    jittered = scaler.inverse_transform((scaled+noise.reshape(-1, 1))).flatten()
    return jittered



# %%
'''
Flips the data
data: shape (N,) univariate time series
to do: add upper and lower bounds?

Reference: Data Augmentation of Wearable Sensor Data for Parkinson’s
Disease Monitoring using Convolutional Neural Networks
''' 
def flip( data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    scaler_flipped = MinMaxScaler(feature_range=(0, 1))
    scaled_flipped = scaler_flipped.fit_transform(np.array(data*(-1)).reshape(-1, 1))
    flipped = scaler.inverse_transform((scaled_flipped.reshape(-1, 1))).flatten()
    
    return flipped



# %%
'''
Flips the data
data: shape (N,) univariate time series
to do: add upper and lower bounds?

Reference: Data Augmentation of Wearable Sensor Data for Parkinson’s
Disease Monitoring using Convolutional Neural Networks
''' 


def trend_scale(data, magnitude, seasonality = 12,upper_bound = None, lower_bound = None):
    
    stl = STL(data, period = seasonality)
    res = stl.fit()
    trend = res.trend

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(np.array(trend).reshape(-1, 1))
    s = scaler.transform(np.array(trend).reshape(-1, 1))

    x = s.flatten()
    x_ = x*magnitude
    
    trend_rotate = scaler.inverse_transform(np.array(x_).reshape(-1, 1)).flatten()
    #trend_rotate = pd.Series(trend_rotate,trend.index)    
    trend_scaled = trend_rotate + res.resid + res.seasonal
    
    if lower_bound != None or upper_bound != None:
        if lower_bound == None:
            lower_bound = np.min(trend_scaled)
        if upper_bound == None:
            upper_bound = np.max(trend_scaled)
        scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))  
        trend_scaled = scaler.fit_transform(np.array(trend_scaled).reshape(-1, 1)).flatten()
       # trend_scaled = pd.Series(trend_scaled,trend.index)
  
    return trend_scaled
  



# %%
'''
Divides the data into N segments and permutes them
data: shape (N,) univariate time series
n_segments: number of segments
n_per: number of permutations, default is 1.

to do: add upper and lower bounds?

Reference: Data Augmentation of Wearable Sensor Data for Parkinson’s
Disease Monitoring using Convolutional Neural Networks
''' 
def permutation(data, n_segments, n_per = 1):
    segments = np.linspace(0,len(data), n_segments+1,dtype= int)
    for per in range(n_per):
        index_segment1 = random.choice(np.arange(1,n_segments+1,1))
        index_segment2 = random.choice(np.arange(1,n_segments+1,1))
        while (index_segment1 == index_segment2):
            index_segment2 = random.choice(np.arange(1,n_segments+1,1))
        
        min_index = min(index_segment1, index_segment2)
        max_index = max(index_segment1, index_segment2)
        
        right = data[segments[max_index]:]
        left = data[:segments[min_index-1]]
        center = data[segments[min_index]:segments[max_index-1]]
        right_per = data[segments[min_index-1]:segments[min_index]]
        left_per = data[segments[max_index-1]:segments[max_index]]
         
        permutated = np.concatenate([left,left_per,center, right_per , right],axis =0)
       
    return permutated


# %%

'''
Divides the data into N segments and permutes them
data: shape (N,) univariate time series
n_segments: number of segments
n_per: number of permutations, default is 1.

to do: add upper and lower bounds?

Reference: Data Augmentation of Wearable Sensor Data for Parkinson’s
Disease Monitoring using Convolutional Neural Networks
''' 
def scale(data, magnitude, upper_bound=None, lower_bound=None):
    return data*magnitude



# %%
def reverse(data):
    r = np.flip(np.array(data))
    return r

'''

data: shape (N,) univariate time series
to do: add upper and lower bounds?

''' 



# %%

# least squares approach - temporal difference
def smooth_ls(data,lambda_):
    length = len(data)
    A = np.zeros((length-1,length))
    for i in range(length-1):
        for j in range(length):
            if i==j:
                A[i,j]=-1
                A[i,j+1]=1
    B = np.eye(length)+lambda_* A.T.dot(A)
    smoothed = np.linalg.lstsq(B,data, rcond=None)[0] 
    return smoothed
    


# %%
# simple exponential smoothing approach

def smooth_ets(data,a):    
    last_y_t = data[0]
    new_y = []
    for x in data:
        y_t = a*x + (1-a)*last_y_t
        new_y.append(y_t)
        last_y_t = y_t
    smoothed =  new_y 
    return smoothed
    

# %%
'''
use MBB to generate new data
data: shape (N,) univariate time series
block_size: the block size for replacing
p: the probability of replacing a block
frequency: the number of samples per unit time 

Reference: Bagging Exponential Smoothing Methods using STL Decomposition and Box-Cox Transformation
''' 
def mbb(data,  p, block_size = None, seasonality = 12 ):
    if p == 0:
        return data

    length = len(data)

    if block_size is None:
        block_size =seasonality*2 if seasonality is not None else min(8, int(length/2))
        
    length = len(data)
    num_blocks = int(length /block_size)
    
    # compute the box-cox transofrmation with adding 1 for avoiding x=0
    data_boxcox1, lambda_ = stats.boxcox(data+1)
    
    stl = STL(data_boxcox1, period = seasonality)
    res = stl.fit()
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid
    
    bs_residual = []
    for i in range(num_blocks + 2):
        x = np.random.uniform()

        # change the remainder block
        if x<p or len(bs_residual)>= length:
            # sample a block from (length - block_size + 1) possible overlapping blocks 
            b_index = random.randint(0, length-block_size)
            bs_residual = np.concatenate((bs_residual, residual[b_index:b_index + block_size ]), axis=0)

        # leave the remainder block unchanged
        else:
            b_index = len(bs_residual)
            orig_block = np.empty((block_size))
            orig_block[:] = np.NaN
            bs_residual = np.concatenate((bs_residual,orig_block), axis=0)
               
    # match the bootstrapped set's length with the original set's length.  
    # so that it won't necessariliy begin or end on a block boundary, we cut a random number of elements 
    # from the start and the remainder from the end.    
    cutout_start = random.randint(0,block_size-1 )

    bs_residual = bs_residual[cutout_start:]  
    cutout_end = len(bs_residual) - length
    bs_residual = bs_residual[:-cutout_end]  
    orig_index = np.where(np.isnan(bs_residual))[0]
    bs_residual[orig_index] = residual[orig_index]    
    mbb_boxcox1 = trend + seasonal + bs_residual

    # reverse the coxbox transformation and adjust to x-1 
    mbb_data  = special.inv_boxcox(mbb_boxcox1,lambda_)-1

    return mbb_data



'''
use gaussian time warping to generate new data
mu: the locaion along the time series to center the mu for the gaussian warping. takes values [0,1]
std: the standard devation of the warping, (0,0.5]
std_boundary: limit the center to a range the includes a std of 1, if mu is not None then ignored.
''' 
def gaussian_tw(data,  std, mu = None, std_boundary = True ):
    index = np.arange(0,len(data),1)
    
    # pick a center for mu
    if mu==None:
        # allow the mu to lie in a range that respects std=1 from both sides,
        # to avoid mu from been too close to the end or start of the series.
        if std_boundary==True and std<0.5:  
            mu = np.random.uniform(low=std, high=1-std) 

        # allow mu to lie in any possible location  within the range.         
        else:                        
            mu = np.random.uniform(low=0, high=1)
    else:
        if mu>1 or mu<0:
            raise Exception("Numbers below 0 and above 1 not accepted")
            
    transform = lambda x: stats.norm.cdf((x/len(index)),mu,std)*len(index)
    vfunc = np.vectorize(transform)
    n_index = np.round(vfunc(index)).astype(int)
    
    # fix boundaries
    n_index[n_index >= len(data)] = len(data)-1
    n_index[n_index <= 0] = 0
    #time_warp= pd.Series(data.values[n_index],data.index)
    time_warp = np.take(data,n_index)
    return time_warp




'''
use dynamic time stretching, this implementation is a little different from that 
proposed in the paper.

max_scale: takes values (0,1], determines the scale magnitude of each window
w: the wraping window size (not overlapping), set to len(data) for a single window



reference: IMPROVING SEQUENCE-TO-SEQUENCE SPEECH RECOGNITION TRAINING WITH
ON-THE-FLY DATA AUGMENTATION
''' 


def dts(data,n_warp, w = None, min_scale = 0.25, max_scale = 4):

    if w is None:      
        n_windows = np.random.choice([7,8,9])
        w = len(data)//n_windows   
    warped = []
    
    # for each window warp
    w_indicies = np.random.choice(len(data)//w+1,size = n_warp,replace=False)
    
    # for each window
    for i in range(len(data)//w+1):     
        window = data[i*w:(i+1)*w]
        
        # if the window was chosen for warping
        if i in w_indicies:            
            # generate a randpom epsilon, if above 0.5 then upscale else downscale
            # then choose a scaling paramater
            eps=np.random.uniform()
            w_mag = np.random.uniform(1,max_scale) if eps>0.5 else np.random.uniform(min_scale,1)
            window = window_warping( window , w_mag)            
        warped = np.concatenate([warped,window])
    return warped
        


'''
warp the the entire time series length

reference: Data Augmentation for Time Series Classification
using Convolutional Neural Networks
''' 
def window_warping(data, mag):
    data = np.array(data)
    size = int(mag*len(data))
    r = np.linspace(0,len(data)-1, size)
    indicies = np.round(r).astype(int)
    warped = np.take(data,indicies)        
    return warped

'''
noise scaling using sobel operator for 1d data.
d: the derivative, 1 for first 2 for second derivatives.
eps: the gap between time stamps
'''
def noise_scale(data, mag, d=1, eps = 1):
    
    if d==1:
        kernel = np.zeros((eps+1))  
        kernel[0] = 1
        kernel[-1] = -1

    if d==2:
        kernel = kernel = np.zeros((2*eps+1))  
        kernel[0] = -1
        kernel[-1] = -1
        kernel[eps] = 2
    noise_component =   np.convolve(data,kernel,mode = 'same') 
    # eliminate the noise  from the first element
    noise_component = np.concatenate([[0],noise_component[1:]]) 
    noise_scaled =  data + mag*noise_component
    return noise_scaled




         
def apply_op(data , op_name: str, magnitude: int, seasonality: int,ranges=ranges):

    if op_name not in ["Flip","Reverse","Identity"]:
        mag =  ranges[op_name][magnitude]

    if op_name == "Jittering":
        output = jittering(data, mag)

    elif op_name == "Flip":
        output = flip(data)
    
    elif op_name == "Trend Scale":
        output = trend_scale(data,mag, seasonality=seasonality)
    
    elif op_name == "Permutation":
        output = permutation(data, mag, n_per = 1)
    
    elif op_name == "Scale":
        output = scale(data,mag)

    elif op_name == "Reverse":
        output = reverse(data)
    
    elif op_name == "Smooth LS":
        output = smooth_ls(data,mag)

    elif op_name == "Smooth ETS":
        output = smooth_ets(data,mag)
    
    elif op_name == "Gaussian TW":
        output = gaussian_tw(data,mag)

    elif op_name == "MBB":
        output = mbb(data, mag, seasonality=seasonality)

    elif op_name == "DTS":
       # mag = len(data)//mag
        output = dts(data,mag)
        
    elif op_name == "Identity":
        output = data
        
    elif op_name == "Window Warping":
        output = window_warping(data, mag)

    elif op_name == "Noise Scale":
        output = noise_scale(data,mag)

    
    return output


def plot_augmentation(data,seasonality,aug_name:str, plots = 8):
    
    data2 = data
    if aug_name in ["Reverse","Flip","Identity"]:
        fig, ax = plt.subplots(1, 2, figsize=(15,3))
        mag = 0
        trans_data = apply_op(data ,aug_name, mag, seasonality, ranges)
        augmented_data = pd.Series(trans_data,index = data.index)
        ax[0].plot(data)
        ax[1].plot( augmented_data, color='orange')

    else:
        mag = ranges[aug_name]
        fig, ax = plt.subplots(int(plots/2), 2, figsize=(15,15))
        
        if aug_name in ["DTS","Gaussian TW","Window Warping"]:
            data= data.values

        for p in range(plots):
            mag = np.arange(0,8)[p]
            trans_data = apply_op(data ,aug_name, mag, seasonality, ranges)  
            
            # different lengths, take out timestamp index
            if aug_name in ["DTS","Gaussian TW","Window Warping"]: 
                augmented_data = trans_data 
                
            # same lengths, use timestamp index
            else:
                augmented_data = pd.Series(trans_data, index = data.index)
                
            if p%2==0:
                ax[int(p/2), 0].plot(data)
                ax[int(p/2), 0].plot(augmented_data, color='orange')
                ax[int(p/2), 0].title.set_text('mag = '+ str(mag))
            else:
                ax[int(p/2), 1].plot(data)
                ax[int(p/2), 1].plot(augmented_data, color='orange')
                ax[int(p/2), 1].title.set_text('mag = '+ str(mag ))
    fig.suptitle(aug_name)
    plt.show()






