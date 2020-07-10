# -*- coding: utf-8 -*-
import numpy as np
import pdb, os

import torch
import torch.nn.functional as F
from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,input):
        return input.view(input.size(0),-1)

class SimpleCNN(nn.Module):
    def __init__(self, num_class=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(32,16, kernel_size=3, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(16)

        # name: "dense.weight", "dense.bias"
        self.dense = nn.Linear(4*4*16, num_class) # 
        self.flatten = Flatten()
        self.softmax = nn.functional.softmax

        self.early_stopper = None

    def forward(self, inputs, hidden=False):
        is_training = self.training

        x = self.conv1(inputs)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # ?, 32, 15, 15

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x) # ?, 16, 4, 4
        
        x = self.flatten(x) # ?, 4*4*16
        out = self.dense(x) 
        out = self.softmax(out, dim=1)

        if hidden:
            return out, x
        else:
            return out

    def predict(self, x, batch_size=1000, onehot=False):
        self.eval()
        total_batch = compute_num_batch(x,batch_size)
        pred_result = []

        for idx in range(total_batch):
            batch_x = x[idx*batch_size:(idx+1)*batch_size]
            pred = self.forward(batch_x)
            if onehot:
                pred_result.append(pred)
            else:
                pred_result.append(torch.argmax(pred,1))

        preds = torch.cat(pred_result)
        return preds



