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

def pm_pd(g):
    pmpd_src_i = th.stack((g.all_edges()[0], th.arange(g.number_of_edges())))
    pmpd_end_i = th.stack((g.all_edges()[1], th.arange(g.number_of_edges())))
    
    pm = th.sparse.FloatTensor(pmpd_src_i, th.ones(g.number_of_edges()), th.Size([g.number_of_nodes(),g.number_of_edges()])).to_dense()
    pd = th.sparse.FloatTensor(pmpd_end_i, th.ones(g.number_of_edges()), th.Size([g.number_of_nodes(),g.number_of_edges()])).to_dense()
    
    return pm, pd

class GNNModule(nn.Module):
    def __init__(self, in_feats, out_feats, radius, dev):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.radius = radius

        self.dev = dev

        new_linear = lambda: nn.Linear(in_feats, out_feats // 2)

        new_linear_ = lambda: nn.Linear(out_feats, out_feats // 2)

        self.theta = new_linear()

        ## Start relu part
        self.theta_r = new_linear()

        ## End relu part

        self.bn_x = nn.BatchNorm2d(out_feats)

    def forward(self, x0, x1, deg_g, g_a1):

        x_r = self.theta_r(th.mm(g_a1, x1) + (deg_g - 1) * x0)
        x = self.theta(th.mm(g_a1, x1) + (deg_g - 1) * x0)

        x = th.cat([x, F.relu(x_r)], 1)
        x = self.bn_x(x.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        return x1, x

class GNNModule_first(nn.Module):
    def __init__(self, in_feats, out_feats, radius, dev):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.radius = radius

        self.dev = dev

        new_linear = lambda: nn.Linear(in_feats, out_feats // 2)

        self.theta = new_linear()

        ## Start relu part
        self.theta_r = new_linear()

        ## End relu part

        self.bn_x = nn.BatchNorm2d(out_feats)

    def forward(self, x1, g_a1):
        x_r = self.theta_r(th.mm(g_a1, x1))
        x = self.theta(th.mm(g_a1, x1))

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

        self.theta = new_linear()

    def forward(self, x0, x1, deg_g, g_a1):

        x = self.theta(th.mm(g_a1, x1) + (deg_g - 1) * x0)

        return x

class GNN(nn.Module):
    def __init__(self, feats, radius, n_classes, dev):
        """
        Parameters
        ----------
        g : networkx.DiGraph
        """
        super(GNN, self).__init__()
        self.module_first, self.module_second = GNNModule_first(feats[0], feats[1], radius, dev), GNNModule_first(feats[1], feats[2], radius, dev)
        self.module_final = GNNModule_final(feats[-2], feats[-1], radius, dev)
        self.module_list = nn.ModuleList([GNNModule(m, n, radius, dev)
                                          for m, n in zip(feats[2:-2], feats[3:-1])])

    def forward(self, deg_g, g_a1):
        x0 = deg_g

        x0 = self.module_first(x0, g_a1)
        x1 = self.module_second(x0, g_a1)
        
        for module in self.module_list:
            x0, x1 = module(x0, x1, deg_g, g_a1)
        
        x = self.module_final(x0, x1, deg_g, g_a1)
        return x
