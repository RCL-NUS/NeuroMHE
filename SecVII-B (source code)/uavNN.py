"""
This file defines the class of neural network that parameterizes Q and R
------------------------------------------------------------------------
Wang, Bingheng, 02, Jan., 2021, at UTown, NUS
modified on 08, Jan., 2021, at Control & Simulation Lab, NUS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm

class Net(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        # D_in : dimension of input layer
        # D_h  : dimension of hidden layer
        # D_out: dimension of output layer
        super(Net, self).__init__()
        self.linear1 = spectral_norm(nn.Linear(D_in, D_h))
        self.linear2 = spectral_norm(nn.Linear(D_h, D_h))
        self.linear3 = spectral_norm(nn.Linear(D_h, D_out))
        # self.linear1 = nn.Linear(D_in, D_h)
        # self.linear2 = nn.Linear(D_h, D_h)
        # self.linear3 = nn.Linear(D_h, D_out) # linear9
        # self.dropout = nn.Dropout(1e-4)

    def forward(self, input):
        # convert state s to tensor
        S = torch.tensor(input, dtype=torch.float) # column 2D tensor
        z1 = self.linear1(S.t()) # linear function requires the input to be a row tensor
        z2 = F.relu(z1)  # hidden layer 1
        # z2 = self.dropout(z2)
        z3 = self.linear2(z2)
        z4 = F.relu(z3)  # hidden layer 2
        # z4 = self.dropout(z4)
        z5= self.linear3(z4) # output layer
        return z5.t()

    def myloss(self, para, dp):
        # convert np.array to tensor
        Dp = torch.tensor(dp, dtype=torch.float) # row 2D tensor
        loss_nn = torch.matmul(Dp, para)
        return loss_nn





