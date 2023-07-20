"""
This file defines the class of neural network that parameterizes Q and R
------------------------------------------------------------------------
Wang, Bingheng, 02, Jan., 2021, at UTown, NUS
modified on 08, Jan., 2021, at Control & Simulation Lab, NUS
----------------
2nd version on 10 Oct. 2022 after receiving the reviewers' comments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
from torch.nn import init

"""
[1] Bartlett, P.L., Foster, D.J. and Telgarsky, M.J., 2017. 
    Spectrally-normalized margin bounds for neural networks. 
    Advances in neural information processing systems, 30.
"""

class Net(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        # D_in : dimension of input layer
        # D_h  : dimension of hidden layer
        # D_out: dimension of output layer
        super(Net, self).__init__()
        self.linear1 = spectral_norm(nn.Linear(D_in, D_h)) # spectral normalization can stabilize DNN training and help generalization to unseen data well [1]
        self.linear2 = spectral_norm(nn.Linear(D_h, D_h))
        self.linear3 = spectral_norm(nn.Linear(D_h, D_out))

    def forward(self, input):
        # convert state s to tensor
        S = torch.tensor(input, dtype=torch.float) # column 2D tensor
        z1 = self.linear1(S.t()) # linear function requires the input to be a row tensor
        z2 = F.relu(z1)  # hidden layer 1
        z3 = self.linear2(z2)
        z4 = F.relu(z3)  # hidden layer 2
        z5 = self.linear3(z4)  # output layer
        return z5.t()

    def myloss(self, para, dp):
        # convert np.array to tensor
        Dp = torch.tensor(dp, dtype=torch.float) # row 2D tensor
        loss_nn = torch.matmul(Dp, para)
        return loss_nn






