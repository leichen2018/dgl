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

class GNNModule(nn.Module):
    def __init__(self, in_feats, out_feats, radius, dev):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.radius = radius

        self.dev = dev

        new_linear = lambda: nn.Linear(in_feats, out_feats // 2)
        new_linear_list = lambda: nn.ModuleList([new_linear() for i in range(radius)])

        new_linear_ = lambda: nn.Linear(out_feats, out_feats // 2)

        self.theta_x, self.theta_deg = new_linear(), new_linear()
        self.theta_list = new_linear_list()

        ## Start relu part
        self.theta_x_r, self.theta_deg_r = new_linear(), new_linear()
        self.theta_list_r = new_linear_list()
        ## End relu part

        self.bn_x = nn.BatchNorm2d(out_feats)

    def forward(self, x, deg_g, g_a1, g_a2):     
        x_list = []

        x_list.append(th.mm(g_a1, x))
        x_list.append(th.mm(g_a2, x))

        sum_x = sum(theta(z) for theta, z in zip(self.theta_list, x_list))
        sum_x_r = sum(theta(z) for theta, z in zip(self.theta_list_r, x_list))

        x_r = self.theta_x_r(x) + self.theta_deg_r(deg_g * x) + sum_x_r
        x = self.theta_x(x) + self.theta_deg(deg_g * x) + sum_x
        n = self.out_feats // 2
        x = th.cat([x, F.relu(x_r)], 1)
        x = self.bn_x(x.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        return x

class GNNModule_final(nn.Module):
    def __init__(self, in_feats, out_feats, radius, dev):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.radius = radius

        self.dev = dev

        new_linear = lambda: nn.Linear(in_feats, out_feats)
        new_linear_list = lambda: nn.ModuleList([new_linear() for i in range(radius)])

        new_linear_ = lambda: nn.Linear(out_feats, out_feats)

        self.theta_x, self.theta_deg = new_linear(), new_linear()
        self.theta_list = new_linear_list()

    def forward(self, x, deg_g, g_a1, g_a2):

        x_list = []

        x_list.append(th.mm(g_a1, x))
        x_list.append(th.mm(g_a2, x))

        sum_x = sum(theta(z) for theta, z in zip(self.theta_list, x_list))

        x = self.theta_x(x) + self.theta_deg(deg_g * x) + sum_x

        return x

class GNN(nn.Module):
    def __init__(self, feats, radius, n_classes, dev):
        """
        Parameters
        ----------
        g : networkx.DiGraph
        """
        super(GNN, self).__init__()
        self.module_final = GNNModule_final(feats[-2], feats[-1], radius, dev)
        self.module_list = nn.ModuleList([GNNModule(m, n, radius, dev)
                                          for m, n in zip(feats[:-2], feats[1:-1])])

    def forward(self, deg_g, g_a1, g_a2):
        x = deg_g
        
        for module in self.module_list:
            x = module(x, deg_g, g_a1, g_a2)
        
        x = self.module_final(x, deg_g, g_a1, g_a2)
        return x

