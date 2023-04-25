# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:10:08 2021

@author: Anne
"""
import numpy as np
import math

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0) #真实值等于0的地方就不管了
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true - y_pred) / y_true)
    mape[np.isinf(mape)] = 0
    return np.mean(mape) * 100

def WMAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0) #真实值等于0的地方就不管了
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]
    mape1 = np.abs((y_true - y_pred) / y_true)
    mape1[np.isinf(mape1)] = 0
    return sum(mape1*(y_true/sum(y_true))) * 100

def MRE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0) #真实值等于0的地方就不管了
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]
    
    y_true_mean = y_true.mean()
    mre = np.abs((y_true - y_pred) / y_true_mean)
    return np.mean(mre) * 100

def MRE_mask(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 5) 
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]
    
    y_true_mean = y_true.mean()
    mre = np.abs((y_true - y_pred) / y_true_mean)
    return np.mean(mre) * 100

def WMRE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0) 
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]
    
    y_true_mean = y_true.mean()
    mre1 = np.abs((y_true - y_pred) / y_true_mean)
    
    return sum(mre1*(y_true/sum(y_true))) * 100