# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:08:06 2021

@author: Anne
prepare data

"""
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
import numpy as np
import torch 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
plt.rcParams['font.sans-serif'] = ['STSong']
np.seterr(divide='ignore',invalid='ignore')





date =  ['20190225','20190226','20190227','20190228','20190301',
         '20190304','20190305','20190306','20190307','20190308',
         '20190415','20190416','20190417','20190418','20190419']
timelist=[]
for t in date:
    timelist.extend(pd.date_range(start = t +' 0:00:00',
                            periods = 96, freq = '15min'))

def normalization(x):
    # 1336,217,217/1336,1,217：对时间上进行处理让各个时间上的量达到均匀
    x_max = x.max(axis=(0), keepdims=True)
    x_min = x.min(axis=(0), keepdims=True)
    x_norm = (x - x_min) / (x_max - x_min)
    x_norm[np.isnan(x_norm)] = 0
    print(" nan",np.isnan(x_norm).any())
    print(" inf",np.isinf(x_norm).any())
    
    return {'_max': x_max, '_min': x_min}, x_norm


#%% od flow
od_tt = np.load("od.npy").astype(np.int)# (15*96, 217, 217)

od_tt = od_tt.reshape(-1,od_tt.shape[2],od_tt.shape[3]) # (15, 96, 217, 217)

od_tt_norm, od_tt = normalization(od_tt)

od_tt_sum = []
for i in range(len(od_tt)):
    od_tt_sum.append(od_tt[i,:,:].sum())

plt.figure(dpi = 150)
plt.plot(od_tt_sum)
plt.show()

#%% in out flow
in_tt = np.load("in.npy").astype(np.int) # (15*96, 1, 217)
out_tt = np.load("out.npy").astype(np.int) # (15*96, 1, 217)

in_tt_norm, in_tt = normalization(in_tt)
out_tt_norm, out_tt = normalization(out_tt)

out_tt = np.array(out_tt)
in_tt = np.array(in_tt)
in_tt_sum, out_tt_sum= [],[]
for i in range(len(in_tt)):
    in_tt_sum.append(in_tt[i,:].sum())
    out_tt_sum.append(-out_tt[i,:].sum())
plt.figure(dpi = 150)
plt.plot(in_tt_sum)
plt.plot(out_tt_sum)
plt.show()

#%% probability
utility_tt =np.load("utility.npy")


def p_normalization(x):
    # utility to probability
    x = prob_tt.copy()
    x_fz = np.exp(x)
    x_fm = np.sum(np.exp(x),axis = 2,keepdims=True)
    x_p = x_fz/x_fm
    check = x_p.sum(axis=2,keepdims=True)
    check = check- np.ones(check.shape)
    return x_p

prob_tt = p_normalization(utility_tt)

prob_tt = np.expand_dims(prob_tt,0).repeat(len(date),axis = 0).astype(np.float16) # f16
prob_tt = prob_tt.reshape(-1,prob_tt.shape[2],prob_tt.shape[3])


#%% slide windows

slice= 4
tp = 4  # #Tp
tr = 4 * slice # Tr
# feature = 'r' # 'rd'
def search_data_dt(t,dt):
    if t-dt-tp < 0:
        return None,None,None,None
    else:
        xin = in_tt[t-dt-tp:t-dt-tp+tr,:,:]
        xout = out_tt[t-dt-tp:t-dt-tp+tr,:,:]
        xod = od_tt[t-dt:t-dt+tp,:,:]
        
        return xin,xout,xod
    
def search_data_r(t):
    if t-tr < 0:
        return None,None,None
    else:
        xin = in_tt[t-tr:t,:,:]
        xout = out_tt[t-tr:t,:,:]
        return xin,xout
    
x_inin,x_outout = [],[]
x_odod = []
x_probprob = []
y_ppin = []
y_ppout = []
y_ppod = []


for t in range(len(timelist)-tp):

    x_in_r = search_data_r(t)[0]
    x_out_r = search_data_r(t)[1]
    x_od = search_data_dt(t,96)[2]

    if (x_in_r is None) or (x_od is None):
        continue
    else:
        x_in = x_in_r.copy()
        x_out = x_out_r.copy()

    y_pin = in_tt[t:t+tp,:,:]
    y_pout = out_tt[t:t+tp,:,:]
    y_pod = od_tt[t:t+tp,:,:]
    x_prob = prob_tt[t:t+tp,:,:]
    
    if (y_pin is None) or (y_pout is None) or (y_pod is None) or (x_prob is None): 
        print("None")
    
    x_inin.append(x_in)
    x_outout.append(x_out)
    x_odod.append(x_od)
    
    
    y_ppin.append(y_pin)
    y_ppout.append(y_pout)
    y_ppod.append(y_pod)
    
    x_probprob.append(x_prob)

x_inin = np.array(x_inin)
x_outout = np.array(x_outout)
x_odod = np.array(x_odod)

y_ppin= np.array(y_ppin)
y_ppout= np.array(y_ppout)
y_ppod= np.array(y_ppod)
x_probprob = np.array(x_probprob)

print('inout shape:',x_inin.shape,x_outout.shape)
print('od shape:',x_odod.shape)
print('cost shape:',x_probprob.shape)
print('prob shape:',x_probprob.shape)

print('targetin shape:',y_ppin.shape)
print('targetout shape:',y_ppout.shape)
print('targetod shape:',y_ppod.shape)

# %%split
split1 = int(len(y_ppod)*0.6)
split2 =int(len(y_ppod)*0.8)

np.savez_compressed(f'train_iter_tr{tr}_tp{tp}',
                    x_inin[0:split1], x_outout[0:split1],
                    x_odod[0:split1],x_probprob[0:split1],
                    y_ppin[0:split1], y_ppout[0:split1],
                    y_ppod[0:split1])

np.savez_compressed(f'val_iter_tr{tr}_tp{tp}',
                    x_inin[split1:split2], x_outout[split1:split2],
                    x_odod[split1:split2],x_probprob[split1:split2],
                    y_ppin[split1:split2], y_ppout[split1:split2],
                    y_ppod[split1:split2])

np.savez_compressed(f'test_iter_tr{tr}_tp{tp}',
                    x_inin[split2:], x_outout[split2:],
                    x_odod[split2:], x_probprob[split2:],
                    y_ppin[split2:], y_ppout[split2:],
                    y_ppod[split2:])

np.savez_compressed(f'maxmin_tr{tr}_tp{tp}',
                    in_tt_norm, out_tt_norm,
                    od_tt_norm
                    )





