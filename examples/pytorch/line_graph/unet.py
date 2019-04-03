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

class GNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer, self).__init__()
        self.in_feats, self.out_feats = in_feats, out_feats
        self.theta_0, self.theta_1, self.theta_2 = nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)

        nn.init.xavier_uniform(self.theta_0.weight)
        nn.init.xavier_uniform(self.theta_1.weight)
    
    def forward(self, x, A):
        x = self.theta_0(x) + self.theta_1(th.mm(A, x)) + self.theta_2(th.mm(A, th.mm(A, x)))
        x = F.relu(x)
        x = self.bn(x)

        return x

class GNNLayer_final(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GNNLayer_final, self).__init__()
        self.in_feats, self.out_feats = in_feats, out_feats
        self.theta_0, self.theta_1 = nn.Linear(in_feats, out_feats), nn.Linear(in_feats, out_feats)

        #nn.init.xavier_uniform(self.theta_0.weight)
        #nn.init.xavier_uniform(self.theta_1.weight)
    
    def forward(self, x, A):
        x = self.theta_0(x) + self.theta_1(th.mm(A, x))

        return x


class GNNModule(nn.Module):
    def __init__(self, feats_list):
        super(GNNModule, self).__init__()
        feats = [feats_list[0]] + [feats_list[1]] * feats_list[2]
        self.layers = nn.ModuleList([GNNLayer(m, n) for m, n in zip(feats[:-1], feats[1:])])
    
    def forward(self, x, A):
        for layer in self.layers:
            x = layer(x, A)

        return x

class GNNModule_sqrt_softmax(nn.Module):
    def __init__(self, feats_list):
        super(GNNModule_sqrt_softmax, self).__init__()
        feats = [feats_list[0]] + [feats_list[1]] * feats_list[2]
        self.layers = nn.ModuleList([GNNLayer(m, n) for m, n in zip(feats[:-1], feats[1:])])
        self.layer_final = GNNLayer_final(feats[-1], feats[-1])
    
    def forward(self, x, A):
        for layer in self.layers:
            x = layer(x, A)
        
        x = self.layer_final(x, A)

        return F.softmax(x, dim = 1), F.softmax(x, dim = 0)

def pooling(x, pool_tran):
    return th.mm(pool_tran.t(), x)

def unpooling(x, pool_tran):
    return th.mm(pool_tran, x)

def adj_pooling(A, pool_tran):
    return th.mm(th.mm(pool_tran.t(), A), pool_tran)

def adj_unpooling(A, pool_tran):
    return th.mm(th.mm(pool_tran, A), pool_tran.t())

class GNN(nn.Module):
    def __init__(self):
        """
        Parameters
        ----------
        g
        adj
        """
        super(GNN, self).__init__()
        feats_list = [[1, 8, 15], [8, 16, 2], [16, 32, 2], [32, 64, 2], [64, 32, 2], [64, 16, 2], [32, 8, 5], [16, 8, 15]]
        cluster_list = [[8, 100, 5], [16, 32, 5], [32, 8, 5]]
        self.embd_gnn_list = nn.ModuleList([GNNModule(feats) for feats in feats_list])
        self.pool_gnn_list = nn.ModuleList([GNNModule_sqrt_softmax(cluster) for cluster in cluster_list])
        self.layer_final = GNNLayer_final(8, 5)


    def forward(self, deg, A):
        x = deg

        x = self.embd_gnn_list[0](x, A)
        x_0 = x

        S_0, S_0_1 = self.pool_gnn_list[0](x, A)
        A = adj_pooling(A, S_0)
        x = pooling(x, S_0)

        x = self.embd_gnn_list[1](x, A)
        x_1 = x

        S_1, S_1_1 = self.pool_gnn_list[1](x, A)
        A = adj_pooling(A, S_1)
        x = pooling(x, S_1)

        x = self.embd_gnn_list[2](x, A)
        x_2 = x

        S_2, S_2_1 = self.pool_gnn_list[2](x, A)
        A = adj_pooling(A, S_2)
        x = pooling(x, S_2)

        x = self.embd_gnn_list[3](x, A)
        
        A = adj_unpooling(A, S_2_1)
        x = unpooling(x, S_2_1)
        
        x = self.embd_gnn_list[4](x, A)
        x = th.cat([x_2, x], dim = 1)

        A = adj_unpooling(A, S_1_1)
        x = unpooling(x, S_1_1)

        x = self.embd_gnn_list[5](x, A)
        x = th.cat([x_1, x], dim = 1)

        A = adj_unpooling(A, S_0_1)
        x = unpooling(x, S_0_1)

        x = self.embd_gnn_list[6](x, A)
        x = th.cat([x_0, x], dim = 1)

        x = self.embd_gnn_list[7](x, A)

        x = self.layer_final(x, A)

        return x

        
