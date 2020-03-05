#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# In[10]:


class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,dilation,dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,                                         stride=stride,padding=(padding,0),dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = weight_norm(nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,                                         stride=stride,padding=(padding,0),dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.residual = nn.Sequential(self.conv1,self.relu1,self.dropout1,                                      self.conv2,self.relu2,self.dropout2)
        self.shortcut = nn.Sequential()
        if in_channels!=out_channels:
            self.shortcut = nn.Conv1d(in_channels,out_channels,kernel_size)

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.conv1.weight.data, 0, 0.01)
        nn.init.normal_(self.conv2.weight.data, 0, 0.01)
        nn.init.normal_(self.shortcut.weight.data, 0, 0.01)
            
    def forward(self,x):
        residual = self.residual(x)
        shortcut = self.short(x)
        return nn.ReLU()(residual+shortcut)


# In[16]:


class TemporalConvNet(nn.Module):
    def __init__(self,num_inputs,num_channels,kernel_size=2,dropout=0.2):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        in_channels = [num_inputs] + num_channels
        
        for i in range(num_levels):
            dilation = 2**i
            layers.append(BasicBlock(in_channels[i],in_channels[i+1],kernel_size,stride=1,dilation=dilation,                                   padding=(kernel_size-1)*dilation,dropout=dropout))
        self.network = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.network(x)


# In[17]:


TemporalConvNet(8,[4,2,1])

