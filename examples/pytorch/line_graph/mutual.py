import copy
import itertools
import dgl
import dgl.function as fn
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class GNNLayer_simple(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer_simple, self).__init__()
        self.in_feats, self.out_feats = in_feats, out_feats
        self.theta_0, self.theta_1, self.theta_2, self.theta_3 = \
            nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
    
    def forward(self, x, A, deg):
        x = self.theta_0(x) + self.theta_1(th.mm(A, x)) + self.theta_2(th.mm(A, th.mm(A, x))) + self.theta_3(deg * x)
        x = F.relu(x)
        x = self.bn(x)

        return x

class GNNLayer_final(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer_final, self).__init__()
        self.in_feats, self.out_feats = in_feats, out_feats
        self.theta_0, self.theta_1= \
            nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats)
    
    def forward(self, x, A):
        x = self.theta_0(x) + self.theta_1(th.mm(A, x))

        return x

class GNNModule_tran(nn.Module):
    def __init__(self, in_feats, out_feats, layers):
        super(GNNModule_tran, self).__init__()
        feats = [in_feats] + [out_feats] * layers
        self.layer_list = nn.ModuleList([GNNLayer_simple(m, n) for m, n in zip(feats[:-1], feats[1:])])
        self.layer_final = GNNLayer_final(feats[-1], feats[-1])
    
    def forward(self, x, A, deg):
        for layer in self.layer_list:
            x = layer(x, A, deg)

        x = self.layer_final(x, A)

        return F.softmax(x, dim = 1), F.softmax(x, dim = 0)

class GNNLayer_mutual(nn.Module):
    def __init__(self, x_feats, y_feats):
        super(GNNLayer_mutual, self).__init__()
        self.x_in_feat, self.x_out_feat = x_feats[0], x_feats[1]
        self.y_in_feat, self.y_out_feat = y_feats[0], y_feats[1]
        
        new_linear_xx = lambda: nn.Linear(self.x_in_feat, self.x_out_feat)
        new_linear_xy = lambda: nn.Linear(self.x_in_feat, self.y_out_feat)
        new_linear_yx = lambda: nn.Linear(self.y_in_feat, self.x_out_feat)
        new_linear_yy = lambda: nn.Linear(self.y_in_feat, self.y_out_feat)

        self.theta_x_0, self.theta_x_1, self.theta_x_2 = new_linear_xx(), new_linear_xx(), new_linear_xx()
        self.theta_x_y = new_linear_yx()

        self.theta_y_0, self.theta_y_1, self.theta_y_2 = new_linear_yy(), new_linear_yy(), new_linear_yy()
        self.theta_y_x = new_linear_xy()

        self.bn_x = nn.BatchNorm1d(self.x_out_feat)
        self.bn_y = nn.BatchNorm1d(self.y_out_feat)

        self.theta_x_3 = new_linear_xx()
        self.theta_y_3 = new_linear_yy()

    def forward(self, x, y, A_x, A_y, S_0, S_1, deg_x, deg_y):
        #y = self.theta_y_0(y) + self.theta_y_1(th.mm(A_y, y)) + self.theta_y_x(th.mm(S_0.t(), x))

        y = self.theta_y_0(y) + self.theta_y_1(th.mm(A_y, y)) + self.theta_y_2(th.mm(A_y, th.mm(A_y, y))) + self.theta_y_x(th.mm(S_0.t(), x)) + \
            self.theta_y_3(deg_y * y)
        y = F.relu(y)
        y = self.bn_y(y)

        #x = self.theta_x_0(x) + self.theta_x_1(th.mm(A_x, x)) + self.theta_x_y(th.mm(S_1, y))
        x = self.theta_x_0(x) + self.theta_x_1(th.mm(A_x, x)) + self.theta_x_2(th.mm(A_x, th.mm(A_x, x))) + self.theta_x_y(th.mm(S_1, y)) + \
            self.theta_x_3(deg_x * x)
        x = F.relu(x)
        x = self.bn_x(x)

        return x, y

class GNN(nn.Module):
    def __init__(self, n_clusters):
        super(GNN, self).__init__()
        tran_layers = 10
        self.tran_module = GNNModule_tran(1, n_clusters, tran_layers)
        self.mainstream_module = nn.ModuleList([GNNLayer_mutual([8, 8], [8, 8]) for i in range(30)])
        self.layer_first = nn.ModuleList([GNNLayer_simple(1, 8) for i in range(2)])
        self.layer_final = GNNLayer_final(8, 5)

    def forward(self, deg_g, A_x):
        x = deg_g

        S_0, S_1 = self.tran_module(x, A_x, deg_g)
        y = th.mm(S_0.t(), x)
        A_y = th.mm(S_0.t(), th.mm(A_x, S_0))
        deg_y = th.sum(A_y, dim = 1, keepdim = True)

        x, y = self.layer_first[0](x, A_x, deg_g), self.layer_first[1](y, A_y, deg_y)
        
        for layer in self.mainstream_module:
            x, y = layer(x, y, A_x, A_y, S_0, S_0, deg_g, deg_y)
        
        x = self.layer_final(x, A_x)

        return x 
        