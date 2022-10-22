import torch as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import math
from LSTMs.Lstm_interface import LstmHyperNetInterface
from mnets.mnet_interface import MainNetInterface


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        '''  
        self.x2h= nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)
        '''
        self.x2h_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.x2h_f = nn.Linear(input_size, hidden_size, bias=bias)
        self.x2h_c = nn.Linear(input_size, hidden_size, bias=bias)
        self.x2h_o = nn.Linear(input_size, hidden_size, bias=bias)
        print('self.x2h', self.x2h_i)
        self.h2h_i = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.h2h_f = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.h2h_c = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.h2h_o = nn.Linear(hidden_size, hidden_size, bias=bias)
        print('self.h2h', self.h2h_i)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden, task_id, theta=None, test=False):
        hx, cx = hidden
        x = x.view(-1, x.size(1))

        if task_id > 0:
            """
            Hidden state weights of LSTM are frozen after first task.
            """
            self.h2h_i.weight.requires_grad = False
            self.h2h_f.weight.requires_grad = False
            self.h2h_c.weight.requires_grad = False
            self.h2h_o.weight.requires_grad = False
            self.h2h_i.bias.requires_grad = False
            self.h2h_f.bias.requires_grad = False
            self.h2h_c.bias.requires_grad = False
            self.h2h_o.bias.requires_grad = False
        if not test:
            ingate = self.x2h_i(x) + self.h2h_i(hx)
            forgetgate = self.x2h_f(x) + self.h2h_f(hx)
            cellgate = self.x2h_c(x) + self.h2h_c(hx)
            outgate = self.x2h_o(x) + self.h2h_o(hx)
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)
            cy = F.mul(cx, forgetgate) + F.mul(ingate, cellgate)
            hy = F.mul(outgate, F.tanh(cy))
        if test:
            """
            For testing we need to use the paramters that are stored during training for network growth technique in which. 
            In each thata hidden parameters are frozen
            """
            self.x2h_i1 = F.matmul(x, (F.transpose(theta[0], 0, 1))) + theta[1]
            self.x2h_f1 = F.matmul(x, (F.transpose(theta[2], 0, 1))) + theta[3]
            self.x2h_c1 = F.matmul(x, (F.transpose(theta[4], 0, 1))) + theta[5]
            self.x2h_o1 = F.matmul(x, (F.transpose(theta[6], 0, 1))) + theta[7]
            self.h2h_i1 = F.matmul(hx, (F.transpose(theta[8], 0, 1))) + theta[9]
            self.h2h_f1 = F.matmul(hx, (F.transpose(theta[10], 0, 1))) + theta[11]
            self.h2h_c1 = F.matmul(hx, (F.transpose(theta[12], 0, 1))) + theta[13]
            self.h2h_o1 = F.matmul(hx, (F.transpose(theta[14], 0, 1))) + theta[15]
            ingate = self.x2h_i1 + self.h2h_i1
            forgetgate = self.x2h_f1 + self.h2h_f1
            cellgate = self.x2h_c1 + self.h2h_c1
            outgate = self.x2h_o1 + self.h2h_o1
            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)
            cy = F.mul(cx, forgetgate) + F.mul(ingate, cellgate)
            hy = F.mul(outgate, F.tanh(cy))

        return hy, cy
