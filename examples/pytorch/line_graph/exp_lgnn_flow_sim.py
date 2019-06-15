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

    deg_end_inverse = g.in_degrees(g.all_edges()[1]).float() - 1
    deg_end_inverse = th.where(deg_end_inverse>0, 1/deg_end_inverse, th.zeros(deg_end_inverse.size()))
    
    pm = th.sparse.FloatTensor(pmpd_src_i, th.ones(g.number_of_edges()), th.Size([g.number_of_nodes(),g.number_of_edges()])).to_dense()
    #pd = th.sparse.FloatTensor(pmpd_end_i, th.ones(g.number_of_edges()), th.Size([g.number_of_nodes(),g.number_of_edges()])).to_dense()
    pd = th.sparse.FloatTensor(pmpd_end_i, deg_end_inverse, th.Size([g.number_of_nodes(), g.number_of_edges()])).to_dense()

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

        self.theta_a, self.theta_deg, self.theta_y = new_linear(), new_linear(), new_linear()

        self.gamma_a, self.gamma_x = new_linear(), new_linear_()

        ## Start relu part
        self.theta_a_r, self.theta_deg_r, self.theta_y_r = new_linear(), new_linear(), new_linear()

        self.gamma_a_r, self.gamma_x_r = new_linear(), new_linear_()
        ## End relu part

        self.bn_x = nn.BatchNorm2d(out_feats)
        self.bn_y = nn.BatchNorm2d(out_feats)

    def forward(self, x0, x1, y, deg_g, g_a1, lg_a1, pm, pd):
        pd_y = th.mm(pd, y)

        #x_r = self.theta_a_r(th.mm(g_a1, x1)) + self.theta_deg_r((deg_g - 1) * x0) + self.theta_y_r(pd_y)
        #x = self.theta_a(th.mm(g_a1, x1)) + self.theta_deg((deg_g - 1) * x0) + self.theta_y(pd_y)
        ##x_r = self.theta_a_r(th.mm(g_a1, x1)) + self.theta_y_r(pd_y)
        ##x = self.theta_a(th.mm(g_a1, x1)) + self.theta_y(pd_y)
        x_r = self.theta_y_r(pd_y)
        x = self.theta_y(pd_y)

        x = th.cat([x, F.relu(x_r)], 1)
        x = self.bn_x(x.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        pm_x = th.mm(pm.t(), x)

        y_r = self.gamma_a_r(th.mm(lg_a1, y)) + self.gamma_x_r(pm_x)
        y = self.gamma_a(th.mm(lg_a1, y)) + self.gamma_x(pm_x)
        y = th.cat([y, F.relu(y_r)], 1)
        y = self.bn_y(y.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        return x1, x, y

class GNNModule_first(nn.Module):
    def __init__(self, in_feats, out_feats, radius, dev):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.radius = radius

        self.dev = dev

        new_linear = lambda: nn.Linear(in_feats, out_feats // 2)

        new_linear_ = lambda: nn.Linear(out_feats, out_feats // 2)

        self.theta_a, self.theta_y = new_linear(), new_linear()

        self.gamma_a, self.gamma_x = new_linear(), new_linear_()

        ## Start relu part
        self.theta_a_r, self.theta_y_r = new_linear(), new_linear()

        self.gamma_a_r, self.gamma_x_r = new_linear(), new_linear_()
        ## End relu part

        self.bn_x = nn.BatchNorm2d(out_feats)
        self.bn_y = nn.BatchNorm2d(out_feats)

    def forward(self, x1, y, g_a1, lg_a1, pm, pd):
        pd_y = th.mm(pd, y)

        ##x_r = self.theta_a_r(th.mm(g_a1, x1)) + self.theta_y_r(pd_y)
        ##x = self.theta_a(th.mm(g_a1, x1)) + self.theta_y(pd_y)
        x_r = self.theta_y_r(pd_y)
        x = self.theta_y(pd_y)

        x = th.cat([x, F.relu(x_r)], 1)
        x = self.bn_x(x.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        pm_x = th.mm(pm.t(), x)

        y_r = self.gamma_a_r(th.mm(lg_a1, y)) + self.gamma_x_r(pm_x)
        y = self.gamma_a(th.mm(lg_a1, y)) + self.gamma_x(pm_x)
        y = th.cat([y, F.relu(y_r)], 1)
        y = self.bn_y(y.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        return x, y

class GNNModule_final(nn.Module):
    def __init__(self, in_feats, out_feats, radius, dev):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.radius = radius

        self.dev = dev

        new_linear = lambda: nn.Linear(in_feats, out_feats)

        new_linear_ = lambda: nn.Linear(out_feats, out_feats)

        self.theta_a, self.theta_deg, self.theta_y = new_linear(), new_linear(), new_linear()

    def forward(self, x0, x1, y, deg_g, g_a1, pm, pd):
        pd_y = th.mm(pd, y)

        #x = self.theta_a(th.mm(g_a1, x1)) + self.theta_deg((deg_g - 1) * x0) + self.theta_y(pd_y)
        ##x = self.theta_a(th.mm(g_a1, x1)) + self.theta_y(pd_y)
        x = self.theta_y(pd_y)

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

    def forward(self, deg_g, deg_lg, g_a1, lg_a1, pm, pd):
        x0, y = deg_g, deg_lg

        x0, y = self.module_first(x0, y, g_a1, lg_a1, pm, pd)
        x1, y = self.module_second(x0, y, g_a1, lg_a1, pm, pd)
        
        for module in self.module_list:
            x0, x1, y = module(x0, x1, y, deg_g, g_a1, lg_a1, pm, pd)
        
        x = self.module_final(x0, x1, y, deg_g, g_a1, pm, pd)
        return x
