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
    
    pm = th.sparse.FloatTensor(th.cat((pmpd_src_i, pmpd_end_i), 1), th.ones(2*g.number_of_edges()), th.Size([g.number_of_nodes(),g.number_of_edges()])).to_dense()
    pd = th.sparse.FloatTensor(th.cat((pmpd_src_i, pmpd_end_i), 1), th.cat((th.ones(g.number_of_edges()), th.ones(g.number_of_edges())*-1), 0), th.Size([g.number_of_nodes(), g.number_of_edges()])).to_dense()
    
    return pm, pd

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

        self.theta_x, self.theta_deg, self.theta_y_0, self.theta_y_1 = \
            new_linear(), new_linear(), new_linear(), new_linear()
        self.theta_list = new_linear_list()

        self.gamma_y, self.gamma_deg = \
            new_linear(), new_linear()
        self.gamma_list = new_linear_list()

        self.gamma_x_0, self.gamma_x_1 = new_linear_(), new_linear_()

        ## Start relu part
        self.theta_x_r, self.theta_deg_r, self.theta_y_0_r, self.theta_y_1_r = \
            new_linear(), new_linear(), new_linear(), new_linear()
        self.theta_list_r = new_linear_list()

        self.gamma_y_r, self.gamma_deg_r = \
            new_linear(), new_linear()
        self.gamma_list_r = new_linear_list()

        self.gamma_x_0_r, self.gamma_x_1_r = new_linear_(), new_linear_()
        ## End relu part

        self.bn_x = nn.BatchNorm2d(out_feats)
        self.bn_y = nn.BatchNorm2d(out_feats)

    def forward(self, x, y, deg_g, deg_lg, g_a1, g_a2, lg_a1, lg_a2, pm, pd):
        pm_y = th.mm(pm, y)
        pd_y = th.mm(pd, y)        

        x_list = []

        x_list.append(th.mm(g_a1, x))
        x_list.append(th.mm(g_a2, x))

        sum_x = sum(theta(z) for theta, z in zip(self.theta_list, x_list))
        sum_x_r = sum(theta(z) for theta, z in zip(self.theta_list_r, x_list))

        x_r = self.theta_x_r(x) + self.theta_deg_r(deg_g * x) + sum_x_r + self.theta_y_0_r(pm_y) + self.theta_y_1_r(pd_y)
        x = self.theta_x(x) + self.theta_deg(deg_g * x) + sum_x + self.theta_y_0(pm_y) + self.theta_y_1(pd_y)
        n = self.out_feats // 2
        x = th.cat([x, F.relu(x_r)], 1)
        #x = F.relu(th.cat([x, x_r], 1))
        x = self.bn_x(x.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        y_list = []
        
        y_list.append(th.mm(lg_a1, y))
        y_list.append(th.mm(lg_a2, y))
        
        sum_y = sum(gamma(z) for gamma, z in zip(self.gamma_list, y_list))
        sum_y_r = sum(gamma(z) for gamma, z in zip(self.gamma_list_r, y_list))

        pm_x = th.mm(pm.t(), x)
        pd_x = th.mm(pd.t(), x)

        y_r = self.gamma_y_r(y) + self.gamma_deg_r(deg_lg * y) + sum_y_r + self.gamma_x_0_r(pm_x) + self.gamma_x_1_r(pd_x)
        y = self.gamma_y(y) + self.gamma_deg(deg_lg * y) + sum_y + self.gamma_x_0(pm_x) + self.gamma_x_1(pd_x)
        y = th.cat([y, F.relu(y_r)], 1)
        #y = F.relu(th.cat([y, y_r], 1))
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
        new_linear_list = lambda: nn.ModuleList([new_linear() for i in range(radius)])

        new_linear_ = lambda: nn.Linear(out_feats, out_feats)

        self.theta_x, self.theta_deg, self.theta_y_0, self.theta_y_1 = \
            new_linear(), new_linear(), new_linear(), new_linear()
        self.theta_list = new_linear_list()

    def forward(self, x, y, deg_g, g_a1, g_a2, pm, pd):
        pm_y = th.mm(pm, y)
        pd_y = th.mm(pd, y)

        x_list = []

        x_list.append(th.mm(g_a1, x))
        x_list.append(th.mm(g_a2, x))

        sum_x = sum(theta(z) for theta, z in zip(self.theta_list, x_list))

        x = self.theta_x(x) + self.theta_deg(deg_g * x) + sum_x + self.theta_y_0(pm_y) + self.theta_y_1(pd_y)

        return x

class GNN(nn.Module):
    def __init__(self, feats, radius, n_classes, dev):
        """
        Parameters
        ----------
        g : networkx.DiGraph
        """
        super(GNN, self).__init__()
        self.module_first = GNNModule(feats[0], feats[1], radius, dev)
        self.module_mid = GNNModule(feats[1], feats[-2], radius, dev)
        self.module_final = GNNModule_final(feats[-2], feats[-1], radius, dev)
        self.num_of_mid = len(feats)-3
        '''
        self.module_list = nn.ModuleList([GNNModule(m, n, radius, dev)
                                          for m, n in zip(feats[:-2], feats[1:-1])])
        '''

    def forward(self, deg_g, deg_lg, g_a1, g_a2, lg_a1, lg_a2, pm, pd):
        x, y = deg_g, deg_lg
        
        x, y = self.module_first(x, y, deg_g, deg_lg, g_a1, g_a2, lg_a1, lg_a2, pm, pd)

        for i in range(self.num_of_mid):
            x, y = self.module_mid(x, y, deg_g, deg_lg, g_a1, g_a2, lg_a1, lg_a2, pm, pd)
        
        x = self.module_final(x, y, deg_g, g_a1, g_a2, pm, pd)
        return x

