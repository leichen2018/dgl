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

        self.theta_x, self.theta_deg, self.theta_a = new_linear(), new_linear(), new_linear()

        ## Start relu part
        self.theta_x_r, self.theta_deg_r, self.theta_a_r = new_linear(), new_linear(), new_linear()

        ## End relu part

        self.bn_x = nn.BatchNorm2d(out_feats)

    def forward(self, x, deg_g, g_a1):

        x_r = self.theta_x_r(x) + self.theta_deg_r(deg_g * x) + self.theta_a_r(th.mm(g_a1, x))
        x = self.theta_x(x) + self.theta_deg(deg_g * x) + self.theta_a(th.mm(g_a1, x))

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

        self.theta_x, self.theta_deg, self.theta_a = new_linear(), new_linear(), new_linear()

    def forward(self, x, deg_g, g_a1):

        x = self.theta_x(x) +  self.theta_deg(deg_g * x) + self.theta_a(th.mm(g_a1, x))

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

    def forward(self, deg_g, g_a1):
        x = th.ones(deg_g.size()).to('cuda:0')
        
        for module in self.module_list:
            x = module(x, deg_g, g_a1)
        
        x = self.module_final(x, deg_g, g_a1)
        return x

