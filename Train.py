# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 18:25:51 2021

@author: Anne
"""
import torch
import torch.nn as nn
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#%% Load Dataset
# path = '/data/home/u20114054/STNN-ODDP/'
from torch.utils import data
path  = '/home/test/PycharmProjects/Anne'
# path = 'C:/Users/anne_/OneDrive - bjtu.edu.cn/1 OD动态估计'
os.chdir(path)

DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")



if torch.cuda.is_available():
    torch.cuda.empty_cache()

#TODO :configuration
Tr, Tp = 16,4
Feature = 2
N =217
filters = 32
alpha = 1
batch = 32
lr = 0.001
adj = "Lspace"
# adj = "ODweight"
mask = 'mask'
zoom='zoom'
dropout = 0.

epoches = 200

model = "DCM"


train = np.load(f'data/train_iter_tr{Tr}_tp{Tp}.npz',allow_pickle=True)
trainX1 = torch.tensor(train[train.files[0]]).to(torch.float32)#in2
trainX2 = torch.tensor(train[train.files[1]]).to(torch.float32) #out2
trainX = torch.concat((trainX1,trainX2),axis = 2).to(DEVICE)

trainX3 = torch.tensor(train[train.files[2]]).to(torch.float32).to(DEVICE)#od1

trainX4 = torch.tensor(train[train.files[3]]).to(torch.float32).to(DEVICE)# cost1
trainX5 = torch.tensor(train[train.files[4]]).to(torch.float32).to(DEVICE)# prob

trainY1 = torch.tensor(train[train.files[5]]).to(torch.float32).to(DEVICE)# pre_in
# trainY2 = torch.tensor(train[train.files[6]]).to(torch.float32).to(DEVICE) # pre_out
trainY3 = torch.tensor(train[train.files[7]]).to(torch.float32).to(DEVICE) # pre_od

train_set = data.TensorDataset(trainX,trainX3,trainX4, trainX5,
                               trainY1, trainY3) # 
train_iter = data.DataLoader(train_set, batch_size=batch, shuffle = False)
del trainX,trainX3,trainX4,trainX5, trainY3

val = np.load(f'data/val_iter_tr{Tr}_tp{Tp}.npz',allow_pickle=True)
valX1 = torch.tensor(val[val.files[0]]).to(torch.float32)
valX2 = torch.tensor(val[val.files[1]]).to(torch.float32)
valX = torch.concat((valX1,valX2),axis = 2).to(DEVICE)
valX3 = torch.tensor(val[val.files[2]]).to(torch.float32).to(DEVICE)
valX4 = torch.tensor(val[val.files[3]]).to(torch.float32).to(DEVICE)
valX5 = torch.tensor(val[val.files[4]]).to(torch.float32).to(DEVICE)
valY1 = torch.tensor(val[val.files[5]]).to(torch.float32).to(DEVICE)
valY3 = torch.tensor(val[val.files[7]]).to(torch.float32).to(DEVICE)
val_set = data.TensorDataset(valX, valX3, valX4,valX5,valY1, valY3)
val_iter = data.DataLoader(val_set, batch_size=batch, shuffle = False)
del valX,valX3,valX4, valX5, valY3

maxmin = np.load(f"data/maxmin_tr{Tr}_tp{Tp}.npz",allow_pickle = True)

in_min = maxmin[maxmin.files[0]].tolist()["_min"]
in_max = maxmin[maxmin.files[0]].tolist()["_max"]
od_min =  maxmin[maxmin.files[2]].tolist()["_min"]
od_max =  maxmin[maxmin.files[2]].tolist()["_max"]
# Ztemp = in_max/od_max


if zoom == 'zoom':
    Z = in_max.transpose(0,2,1)/od_max
    Z[np.isinf(Z)] = 1
    Z = torch.from_numpy(Z.astype(np.float32))     # 放大系数
elif zoom == 'nozoom':
    Z = torch.from_numpy(np.ones((N,N)))
Z = Z.to(DEVICE)

xdi = torch.from_numpy(np.ones((N,N))).to(DEVICE)

adj_mx = np.load(f'data/adj_{adj}.npy').astype(np.float32)

if mask == 'mask':
    mask_mx = torch.tensor(np.load("data/mask.npy")).to(DEVICE)
else:
    mask_mx = torch.tensor(np.ones((N,N))).to(DEVICE)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
#%% train 
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

# todo :
from ODMP import make_model
print('import model', model)

import torch.optim as optim
from time import time

net = None
net = make_model(DEVICE, adj_mx, Tr,Tp, Feature, N, filters, Z, dropout).to(DEVICE)

if torch.cuda.is_available():
    net = nn.DataParallel(net)
    net.cuda()
# print(net)

# TODO: loss
criterion = nn.MSELoss(reduction='none').to(DEVICE)
# criterion = nn.L1Loss(reduction='none').to(DEVICE)
optimizer = optim.Adam(net.parameters(), lr=lr)

experience = f'experience_new2_{model}'
print(experience)

if not os.path.exists(experience):
    print("makedir")
    os.makedirs(experience)
    start_epoch = 0
else:
    files = os.listdir(path+'/'+experience)
    files = list(filter(lambda x: 'epoch_' in x , files))
    best_epoch = max([int(x[6:]) for x in files])
    if best_epoch > 1:
        start_epoch = best_epoch
        filename = experience +'/epoch_%s' % start_epoch
        net.load_state_dict(torch.load(filename))
        print("load model:", filename)
    else:
        start_epoch = 0
best_val_loss = np.inf

loss_log_all =[]
validation_loss_log_all=[]

start_time = time()
for epoch in range(start_epoch+1, epoches): # 200
    print("epoch: %d, time:%.2f"%(epoch,time()-start_time))
    
    with torch.no_grad():
        val_loss_log = []  # 记录了所有batch的loss
        val_prediction,val_target = [],[] # 存储所有batch的output
        for xinout, xod, xut, prob, yin, yod in val_iter:

            yin = yin.permute(0,1,3,2)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if model == "branch2":
                yp_od = net.module.branch2(xod)
            elif model == "direct":
                yp_od = net(xinout,xod,xdi)
            elif model == "prob":
                yp_od = net(xinout,xod,prob)
            elif model == "DCM": 
                yp_od = net(xinout,xod,xut)

            val_loss_od = (mask_mx * criterion(yp_od, yod)).mean()
            val_loss = val_loss_od
            val_loss_log.append(val_loss.item())
            

        validation_loss = sum(val_loss_log) / len(val_loss_log)
        validation_loss_log_all.append(validation_loss)
        print(' validation loss: %.4f' % (validation_loss))
        # break

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        best_epoch = epoch
        if isinstance(net, nn.DataParallel):
            torch.save(net.module.state_dict(), experience + '/epoch_%s' % best_epoch)
        else:
            torch.save(net.state_dict(), experience + '/epoch_%s' % best_epoch)
    net.train()
    train_loss_log = []
    train_prediction,train_target = [],[]
    for xinout, xod, xut, prob,yin, yod in train_iter:

        yin = yin.permute(0,1,3,2)
        
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if model == "branch2":
            yp_od = net.module.branch2(xod)
            loss = (mask_mx * criterion(yp_od, yod)).mean()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_loss_log.append(train_loss)
        elif model == "direct":
            yp_in = net.module.branch1(xinout)
            yp_od = net(xinout,xod,xdi)

            loss_in = (mask_mx * criterion(yp_in, yin)).mean()
            loss_in.backward(retain_graph=True)    
            loss = (mask_mx * criterion(yp_od, yod)).mean()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_loss_log.append(train_loss)
            
        elif model == "prob":
            yp_in = net.module.branch1(xinout)
            yp_od = net(xinout,xod,prob)

            loss_in = (mask_mx * criterion(yp_in, yin)).mean()
            loss_in.backward(retain_graph=True)    
            loss = (mask_mx * criterion(yp_od, yod)).mean()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_loss_log.append(train_loss)
            
        elif model == "DCM": 
            yp_in = net.module.branch1(xinout)
            yp_od = net(xinout,xod,xut)
            
            loss_in = (mask_mx * criterion(yp_in, yin)).mean()
            loss_in.backward(retain_graph=True)    
            loss = (mask_mx * criterion(yp_od, yod)).mean()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_loss_log.append(train_loss)
   
    train_loss = sum(train_loss_log) / len(train_loss_log )
    loss_log_all.append(train_loss)
    print(' train loss: %.4f' % (train_loss))
    


print('Model',experience)
print('best_epoch',best_epoch)
import matplotlib.pyplot as plt
# plt.plot(loss_log_all[0:-3])
plt.plot(loss_log_all[1:],label = 'train')
plt.plot(validation_loss_log_all[1:], label = 'validation')
plt.title('loss curve')
plt.legend()
plt.savefig(experience+"/losscurve.jpg")

# %% test 
test = np.load(f'data/test_iter_tr{Tr}_tp{Tp}.npz',allow_pickle=True)

testX1 = torch.tensor(test[test.files[0]]).to(torch.float32)
testX2 = torch.tensor(test[test.files[1]]).to(torch.float32)
testX = torch.concat((testX1,testX2),axis = 2).to(DEVICE)

testX3 = torch.tensor(test[test.files[2]]).to(torch.float32).to(DEVICE)
testX4 = torch.tensor(test[test.files[3]]).to(torch.float32).to(DEVICE)
testX5 = torch.tensor(test[test.files[4]]).to(torch.float32).to(DEVICE)

testY1 = torch.tensor(test[test.files[5]]).to(torch.float32).to(DEVICE)
testY3 = torch.tensor(test[test.files[7]]).to(torch.float32).to(DEVICE)

test_set = data.TensorDataset(testX, testX3, testX4,testX5, testY1, testY3) 
test_iter = data.DataLoader(test_set, batch_size=batch, shuffle = False)

mask_mx = torch.tensor(np.load("data/mask.npy")).to(DEVICE)
if isinstance(net, nn.DataParallel):
    net.module.load_state_dict(torch.load(experience+'/epoch_%s'%best_epoch))
else:
    net.load_state_dict(torch.load(experience+'/epoch_%s'%best_epoch))
    
net.train(False) # ensure dropout layers are in test mode
with torch.no_grad():
    prediction_in = []
    target_in =[]
    prediction_inod = []
    prediction_hod = []
    prediction_od = []
    target_od =[]
    
    for xinout, xod, xut, prob, yin, yod in test_iter:
        
        yin = yin.permute(0,1,3,2)
       
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if model == "branch2":
            yp_od = net.branch2(xod)
        elif model == "direct":
            yp_od = net(xinout, xod, xdi)
        elif model == "prob":
            yp_od = net(xinout, xod, prob)
        elif model == "DCM":
            yp_od = net(xinout, xod, xut)

        prediction_od.append(yp_od)
        target_od.append(yod)


    prediction_od = torch.cat(prediction_od,dim=0)
    target_od = torch.cat(target_od,dim=0)
    prediction_od = prediction_od.detach().cpu().numpy()
    target_od = target_od.detach().cpu().numpy()
    prediction_od[prediction_od<0]=0
    print('prediction:', prediction_od.shape)
    print('target:', target_od.shape)

#%% 误差

def re_normalization(x,min,max):
    return min + x * (max - min)


re_prediction = re_normalization(prediction_od, od_min, od_max)
re_target = re_normalization(target_od, od_min, od_max)



from Metrics import MAPE,WMAPE,MRE,WMRE
from sklearn.metrics import mean_squared_error as MSE#均方误差
from sklearn.metrics import mean_absolute_error as MAE

for pre in [(re_prediction,re_target)]:
    re_prediction = pre[0]
    re_target = pre[1]

    mae = MAE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1))
    rmse = MSE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1)) ** 0.5
    mape = MAPE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1))
    mre = MRE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1))
    wmape = WMAPE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1))
    wmre = WMRE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1)) # 不要采用，比MRE高
    


    print('all RMSE: %.3f' % (rmse))
    print('all MAE: %.3f' % (mae))
    print('all MRE: %.3f' % (mre))
    
    mask_mx = mask_mx.detach().cpu().numpy()
    re_prediction = pre[0] * mask_mx
    re_target = pre[1]  * mask_mx

    mae = MAE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1))
    rmse = MSE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1)) ** 0.5
    mape = MAPE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1))
    mre = MRE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1))
    wmape = WMAPE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1))
    wmre = WMRE(re_target.reshape(-1, 1), re_prediction.reshape(-1, 1)) # 不要采用，比MRE高
    
    
    print('mask RMSE: %.3f' % (rmse))
    print('mask MAE: %.3f' % (mae))
    print('mask MRE: %.3f' % (mre))

    # 45 大石；39珠江新城
    # 49 天河；52 岗顶
    # 21 广州火车站 38广州南站
    
    if re_prediction.shape[2] == 1: #jinzhan 
        for st in [21,38,39,45,49,52]:
            stationpre = re_prediction[:,0,0,st]
            stationtrue = re_target[:,0,0,st]
            plt.figure(dpi=300,figsize=(10, 6))
            plt.plot(stationpre,label='pre')
            plt.plot(stationtrue,label='true')
            plt.legend()
            plt.savefig(experience+f"/{st}.jpg")
    if re_prediction.shape[2] == N:
        for st in [(45,39),(49,52),(21,38)]:
            stationpre = re_prediction[:,0,st[0],st[1]]
            stationtrue = re_target[:,0,st[0],st[1]]
            plt.figure(dpi=300,figsize=(10, 6))
            plt.plot(stationpre,label='pre')
            plt.plot(stationtrue,label='true')
            plt.legend()
            plt.savefig(experience+f"/{st}.jpg")
            
