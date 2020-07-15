# -*- coding: utf-8 -*-
import numpy as np
import pdb, os

import torch
import torch.nn.functional as F
from torch import nn
import math

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,input):
        return input.view(input.size(0),-1)

class SimpleCNN(nn.Module):
    def __init__(self, num_class=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d((2,2))
        # self.bn2 = nn.BatchNorm2d(16)

        # name: "dense.weight", "dense.bias"
        self.linear1 = nn.Linear(64 * 8 * 8, 128) # 
        self.linear2 = nn.Linear(128, 128)
        self.dense = nn.Linear(128, num_class)

        self.flatten = Flatten()

        self.softmax = nn.functional.softmax

        self.early_stopper = None

        self._initialize_weights()
        # for name, param in self.named_parameters():
        #     if "weight" in name:
        #         nn.init.normal_(param, mean=0, std=0.01)
        #     if "bias" in name:
        #         nn.init.constant_(param, val=0.01)

    def forward(self, inputs, hidden=False):
        is_training = self.training

        x = self.conv1(inputs) # ?, 64, 32, 32
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x) # ?, 64, 16, 16

        x = self.conv2(x) # ?, 64, 16, 16
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x) # ?, 128, 8, 8
        
        x = self.flatten(x) # ?, 8*8*128

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

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

    def _initialize_weights(self):
        print("initialize model weights.")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


