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

class Net(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        # D_in : dimension of input layer
        # D_h  : dimension of hidden layer
        # D_out: dimension of output layer
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, D_h)
        self.linear2 = nn.Linear(D_h, D_h)
        self.linear3 = nn.Linear(D_h, D_h)
        self.linear4 = nn.Linear(D_h, D_h)
        self.linear5 = nn.Linear(D_h, D_h)
        self.linear6 = nn.Linear(D_h, D_h)
        self.linear7 = nn.Linear(D_h, D_h)
        self.linear8 = nn.Linear(D_h, D_h)
        self.linear9 = nn.Linear(D_h, D_out)


    def forward(self, input):
        # convert state s to tensor
        S = torch.tensor(input, dtype=torch.float) # column 2D tensor
        z1 = self.linear1(S.t()) # linear function requires the input to be a row tensor
        # Alpha1 = torch.tensor(0.25, dtype=torch.float)
        # alpha1 = nn.Parameter(Alpha1)
        z2 = F.relu(z1)  # hidden layer 1
        # feedback input direcly to the hidden layer 6, together with the ouput of the hidden layer 1
        # Alpha2 = torch.tensor(0.25, dtype=torch.float)
        # alpha2 = nn.Parameter(Alpha2)
        z3 = self.linear2(z2)
        z4 = F.relu(z3)  # hidden layer 2
        # z4_a = torch.cat((z4, S.t()),1)
        # z5 = self.linear3(z4)
        # # Alpha3 = torch.tensor(0.25, dtype=torch.float)
        # # alpha3 = nn.Parameter(Alpha3)
        # z6 = F.relu(z5)  # hidden layer 3
        # z7 = self.linear4(z6)
        # # Alpha4 = torch.tensor(0.25, dtype=torch.float)
        # # alpha4 = nn.Parameter(Alpha4)
        # z8 = F.relu(z7)  # hidden layer 4
        # z9 = self.linear5(z8)
        # Alpha5 = torch.tensor(0.25, dtype=torch.float)
        # alpha5 = nn.Parameter(Alpha5)
        # z10= F.gelu(z9)  # hidden layer 5
        # z11= self.linear6(z10)
        # Alpha6 = torch.tensor(0.25, dtype=torch.float)
        # alpha6 = nn.Parameter(Alpha6)
        # z12= F.gelu(z11) # hidden layer 6
        # z13= self.linear7(z12)
        # Alpha7 = torch.tensor(0.25, dtype=torch.float)
        # alpha7 = nn.Parameter(Alpha7)
        # z14= F.prelu(z13,alpha7) # hidden layer 7
        # z15= self.linear8(z14)
        # Alpha8 = torch.tensor(0.25, dtype=torch.float)
        # alpha8 = nn.Parameter(Alpha8)
        # z16= F.prelu(z15,alpha8) # hidden layer 8
        z7= self.linear9(z4)
        # z17= F.sigmoid(z7)
        # para = torch.softmax(z9) # output layer, row 2D tensor
        return z7.t()

    def myloss(self, para, dp):
        # convert np.array to tensor
        Dp = torch.tensor(dp, dtype=torch.float) # row 2D tensor
        loss_nn = torch.matmul(Dp, para)
        return loss_nn

# class NetCtrl(nn.Module):
#     def __init__(self, D_in, D_h, D_out):
#         # D_in : dimension of input layer
#         # D_h  : dimension of hidden layer
#         # D_out: dimension of output layer
#         super(NetCtrl, self).__init__()
#         self.linear1 = nn.Linear(D_in, D_h)
#         self.linear2 = nn.Linear(D_h, D_h)
#         self.linear3 = nn.Linear(D_h, D_out)

#     def forward(self, input):
#         # convert state s to tensor
#         S = torch.tensor(input, dtype=torch.float) # column 2D tensor
#         z1 = self.linear1(S.t()) # linear function requires the input to be a row tensor
#         Alpha = torch.tensor(0.25, dtype=torch.float)
#         alpha = nn.Parameter(Alpha)
#         z2 = F.prelu(z1, alpha)  # hidden layer
#         z3 = self.linear2(z2)
#         z4 = F.prelu(z3, alpha)  # hidden layer 2
#         z5 = self.linear3(z4)
#         para_gain = torch.sigmoid(z5) # output layer, row 2D tensor
#         return para_gain.t()

#     def myloss(self, para_gain, dg):
#         # convert np.array to tensor
#         Dg = torch.tensor(dg, dtype=torch.float) # row 2D tensor
#         loss_nn = torch.matmul(Dg, para_gain)
#         return loss_nn




