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

        self.theta_x, self.theta_ad, self.theta_ad_2, self.theta_dad = \
            new_linear(), new_linear(), new_linear(), new_linear()

        ## Start relu part
        self.theta_x_r, self.theta_ad_r, self.theta_ad_2_r, self.theta_dad_r = \
            new_linear(), new_linear(), new_linear(), new_linear()
        ## End relu part

        self.bn_x = nn.BatchNorm2d(out_feats)

    def forward(self, x, g_a_d, g_d_a_d):
        x_r = self.theta_ad_r(th.mm(g_a_d, x)) + self.theta_ad_2_r(th.mm(g_a_d, th.mm(g_a_d, x))) + self.theta_dad_r(th.mm(g_d_a_d, x))
        #x_r = self.theta_x_r(x) + self.theta_ad_r(th.mm(g_a_d, x)) + self.theta_dad_r(th.mm(g_d_a_d, x))
        x = self.theta_ad(th.mm(g_a_d, x)) + self.theta_ad_2(th.mm(g_a_d, th.mm(g_a_d, x))) + self.theta_dad(th.mm(g_d_a_d, x))
        #x = self.theta_x(x) + self.theta_ad(th.mm(g_a_d, x)) + self.theta_dad(th.mm(g_d_a_d, x))
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

        self.theta_x, self.theta_ad, self.theta_ad_2, self.theta_dad = \
            new_linear(), new_linear(), new_linear(), new_linear()

    def forward(self, x, g_a_d, g_d_a_d):
        x = self.theta_x(x) + self.theta_ad(th.mm(g_a_d, x)) + self.theta_ad_2(th.mm(g_a_d, th.mm(g_a_d, x))) + self.theta_dad(th.mm(g_d_a_d, x))

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

    def forward(self, g_a_d, g_d_a_d, deg_g):
        x = th.ones(deg_g.size()).to('cuda:0')
        
        for module in self.module_list:
            x = module(x, g_a_d, g_d_a_d)
        
        x = self.module_final(x, g_a_d, g_d_a_d)
        return x

