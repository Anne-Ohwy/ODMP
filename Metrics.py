# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:10:08 2021

@author: Anne
"""
import numpy as np
import math

def MRE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true_mean = y_true.mean()
    mre = np.abs((y_true - y_pred) / y_true_mean)
    return np.mean(mre) * 100
