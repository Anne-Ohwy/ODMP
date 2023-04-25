# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:05:09 2022

@author: Anne

"""

from torch.nn import Parameter
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
import numpy as np
#%% ChanelAttention
class ChanelAttention(nn.Module):
    """
    Squeeze-and-Excitation Networks paper
    SE-Inception Module 
    """
    def __init__(self, chanels, F_N, N):
        super().__init__()
        self.reduction_ratio = 4
        self.num_layers=1
        self.chanels = chanels
        self.globalAvgPool = nn.AvgPool2d((F_N, N), stride=1)
        self.MLP1 = nn.Linear(self.chanels, self.chanels//self.reduction_ratio)
        self.MLP2 = nn.Linear(self.chanels//self.reduction_ratio, self.chanels)
        self.Relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # b,tr(c),f,n
        x.device
        att = self.globalAvgPool(x)
        att = att.view(x.size(0), -1) # (B, C)
        att = self.Relu(self.MLP1(att))
        att = torch.sigmoid(self.MLP2(att))
        att = att.unsqueeze(2). unsqueeze(3).expand_as(x) 
        att_x = att *x
        return att_x
#%% inflow branch

class GraphConvolution(nn.Module):
    def __init__(self, adj_mx, in_features, out_features, device):
        super(GraphConvolution,self).__init__()
        self.K = 3
        self.in_feature = in_features
        self.out_feature = out_features
        self.L_tilde = self.scaled_Laplacian(adj_mx)
        self.cheb_polynomials = self.cheb_polynomial(self.L_tilde, self.K)
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.in_feature, self.out_feature)) for _ in range(self.K)])
        
    def scaled_Laplacian(self, W):  # adj_mx
        # W = adj_mx
        assert W.shape[0] == W.shape[1] 
        D = np.diag(np.sum(W, axis=1)) 
        L = D - W 
        lambda_max = eigs(L, k=1, which='LR')[0].real

        return (2 * L) / lambda_max - np.identity(W.shape[0]) # identity, I
    def cheb_polynomial(self, L_tilde, K):
        N = L_tilde.shape[0]
        cheb_polynomials = [np.identity(N), L_tilde.copy()]
        for i in range(2, K):
            cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        cheb_polynomials = [torch.tensor(_).to(torch.float32) for _ in cheb_polynomials]
        return cheb_polynomials

    def forward(self, x):
        b, tr, f, n =x.shape
        x = x.permute(0,3,2,1) # -> (b,n,f,t)
        outputs = []
        for t in range(tr):
            graph_signal = x[:,:,:,t]
            output = torch.zeros(b, n, self.out_feature).to(x.device)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k].to(x.device)
                theta_k = self.Theta[k].to(x.device) # (fin,fout)
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                # (b,n,f) -> (b,f,n) * (n,n) ->(b,f,n) ->(b,n,f)
                output = output + rhs.matmul(theta_k)
                # (b,n,fout)+(b,n,fin)*(fin,fout)
            outputs.append(output.unsqueeze(-1))
            # (b,n,f,t)
        return torch.cat(outputs, dim=-1).permute(0,3,2,1)
    
            
class Tconv1(nn.Module):
    def __init__(self, in_chanels,out_chanels, Tp, kernel = (1,2)):
        super().__init__()
        self.Tconv = nn.Conv2d(in_chanels, out_chanels, kernel_size=kernel)
        self.Tp = Tp
        self.Relu = nn.ReLU(inplace=True)
    def forward(self, x):

        x = self.Tconv(x)
        _,f,n,t = x.shape
        if t < self.Tp: 
            x_p = x[:,:,:,-(self.Tp-t):]
            x = torch.concat((x_p,x),axis=3)
        x = self.Relu(x)
        return x

class Branch1_block(nn.Module):
    def __init__(self, device, adj, t, tp, in_chanel, out_chanel, N):
        super().__init__()
        self.TAT = ChanelAttention(t, in_chanel, N)
        self.gc1 = GraphConvolution(adj, in_chanel, out_chanel, device)
        self.gc2 = GraphConvolution(adj, out_chanel, out_chanel, device)
        
        self.tc1 = Tconv1(out_chanel, out_chanel, tp)
        self.tc2 = Tconv1(out_chanel, out_chanel, tp)
        self.Relu = nn.ReLU(inplace=True)
        self.ln = nn.LayerNorm(N)
        self.res = nn.Conv2d(in_chanel, out_chanel, (1,1))
        
    def forward(self, x):
        x_init = x
        x = self.TAT(x)
        
        xg = self.gc1(x)
        xg = self.Relu(xg)
        xg = self.gc2(xg)
        xg = self.Relu(xg)
        
        xt = self.tc1(xg.permute(0,2,3,1)) # (b,tr,f,n) -> (b,f,n,tr)
        xt = self.tc2(xt).permute(0,3,1,2) # -> (b,tr,f,n)
        Tshort = xg.size(1) - xt.size(1)
        
        x_res = self.res(x_init[:,Tshort:,:,:].permute(0,2,3,1)).permute(0,3,1,2)# (b,tr,f,n) -> (b,f,n,tr) ->(b,t,f,n)
        x = xt + x_res

        x = self.Relu(self.ln(x))
        
        return x

class Branch1(nn.Module):
    def __init__(self, device, adj, Tr, Tp, feature, N, filter, dropout):
        super().__init__()
        self.block1 = Branch1_block(device, adj, Tr, Tp, feature, filter, N)
        self.block2 = Branch1_block(device, adj,  max(Tr-2,Tp), Tp, filter, filter, N)
        self.fc = nn.Linear(filter*max(Tr-4,Tp),Tp)
        self.Relu=nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.permute(0,3,1,2) # (b,t,f,n) ->(b,n,t,f)
        x = x.reshape(x.size(0),x.size(1),-1) #->(b,n,t*f)
        x = self.fc(x).unsqueeze(3).permute(0,2,1,3) #->(b,n,t,f)->(b,t,n,f)
        x = self.Relu(x)
        x = self.drop(x)
        return x

#%% OD branch

class Sconv(nn.Module):
    def __init__(self, in_chanels,out_chanels, kernel):
        super().__init__()
        self.sconv1 = nn.Conv2d(in_chanels, out_chanels, kernel_size=kernel,
                               padding = 'same')
        self.sconv2 = nn.Conv2d(out_chanels, out_chanels, kernel_size=kernel,
                               padding = 'same')
        self.Relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.Relu(self.sconv1(x))
        x = self.sconv2(x)
        return x

class Branch2_block(nn.Module):
    def __init__(self, device, t, filter, N):
        super().__init__()
        self.TAT = ChanelAttention(t, N, N)
        
        self.sc1 = Sconv(t, filter, kernel=(3,3))
        self.sc2 = Sconv(t, filter, kernel=(5,5))

        self.res = nn.Conv2d(t, filter, (1,1))
        
        self.Relu = nn.ReLU(inplace=True)
        self.ln = nn.BatchNorm2d(filter)

    def forward(self, x):
        x_init = x 
        x = self.TAT(x)     
        xs = self.Relu(self.sc1(x) +self.sc2(x))
        x = self.Relu(self.ln(xs+self.res(x_init)))
        return x

class Branch2(nn.Module):
    def __init__(self, device, Tr, Tp, feature, N, filter, dropout):
        super().__init__()
        self.block1 = Branch2_block(device, Tp, filter, N)
        self.block2 = Branch2_block(device, filter, filter, N)
        self.Relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(filter,Tp)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        xf = self.fc(x.permute(0,2,3,1)).permute(0,3,1,2)
        xf = self.Relu(xf)
        xf = self.drop(xf)
        return xf

#%% fusion
class modelall(nn.Module):
    def __init__(self, device, adj, Tr, Tp, feature, N, filter, Z, dropout):
        super().__init__()
        self.branch1 = Branch1(device, adj, Tr, Tp, feature, N, filter, dropout)
        self.branch2 = Branch2(device, Tr, Tp, feature, N, filter, dropout)
        self.W = nn.Parameter(torch.FloatTensor(2,N,N).to(device))
        self.zoom  = Z # (n,n)
                
    def forward(self, xin, xod, prob):
        # xinout :(B,t,f,n)
        # xod : (B,t,n,n)
        # prob : (B, t, n, n)
        xin = self.branch1(xin)
        xod = self.branch2(xod)
        
        self.zoom = self.zoom.to(xin.device)
        xinod = self.zoom * xin * prob
        
        w1 = torch.exp(self.W[0])/torch.sum(torch.exp(self.W),axis = 0)
        w2 = torch.exp(self.W[1])/torch.sum(torch.exp(self.W),axis = 0)

        xod2 =  w1 * xinod + w2 *xod
        
        return xod2
    

def make_model(device, adj, Tr, Tp, Feature, N, filter, Z, dropout):

    model = modelall(device, adj, Tr, Tp, Feature, N, filter, Z, dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    
    return model

#%% 
if __name__ == '__main__':
    # DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    torch.cuda.manual_seed_all(1)
    
    Tr = 16
    Tp = 4
    N = 217
    feature = 2
    filter = 32

    dropout = 0.
    adj_mx = np.load("data/adj_Lspace.npy").astype(np.float32)
    adj_mx = np.random.random((N,N))
    Z = 1/ torch.randn((N,N))
    net = make_model(DEVICE, adj_mx, Tr, Tp, feature, N, filter, Z, dropout).to(DEVICE)

    xin = torch.randn((32,Tr,feature,N))
    xod = torch.randn((32,Tp,N,N))
    prob = torch.randn((32,Tp,N,N))
    
    y = net(xin, xod, prob)
    
    yin = net.branch1(xin)
