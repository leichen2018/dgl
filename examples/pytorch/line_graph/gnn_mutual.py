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
        self.theta_0, self.theta_1, self.theta_2, self.theta_3 = \
            nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)
    
    def forward(self, x, A, deg):
        x = self.theta_0(x) + self.theta_1(th.mm(A, x)) + self.theta_2(th.mm(A, th.mm(A, x))) + self.theta_3(deg * x)

        return x

class GNN(nn.Module):
    def __init__(self, n_layers, n_classes):
        super(GNN, self).__init__()
        feats = [1] + [8] * n_layers
        self.mainstream_module = nn.ModuleList([GNNLayer_simple(m, n) for m,n in zip(feats[:-1], feats[1:])])
        self.layer_final = GNNLayer_final(8, n_classes)

    def forward(self, deg_g, A_x):
        x = deg_g

        for layer in self.mainstream_module:
            x = layer(x, A_x, deg_g)
        
        x = self.layer_final(x, A_x, deg_g)

        return x 
        