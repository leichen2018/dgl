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

ph = 1000

def real(t):
    for i in range(list(t.size())[0]):
        if (t[i]==-1):
            break
    return i

def real_list(t):
    for i in range(len(t)):
        if t[i] == -1:
            break
    return i

def id_to_m(edges):
    return {'m': edges.src['id']}

def m_to_t(nodes):
    new_t_list = []
    for i in range(nodes.__len__()):
        cur = nodes.data['t'][i]
        
        cur = th.cat((nodes.mailbox['m'][i], cur))
        
        cur = th.unique(cur, sorted=True)

        cur = cur[1:]
        
        cur = th.cat((cur, (th.ones(ph-list(cur.size())[0], dtype=th.long)*-1).to('cuda:0')))
        
        new_t_list.append(cur)
    
    return {'t': th.stack(tuple(new_t_list))}

def m_to_tt(nodes):
    new_t_list = []
    for i in range(nodes.__len__()):
        cur = nodes.data['tt'][i]
        
        cur = th.cat((nodes.mailbox['m'][i][0], cur))
        
        cur = th.unique(cur, sorted=True)

        cur = cur[1:]
        
        cur = th.cat((cur, (th.ones(ph-list(cur.size())[0], dtype=th.long)*-1).to('cuda:0')))
        
        new_t_list.append(cur)
        
    return {'tt': th.stack(tuple(new_t_list))}

def t_to_m(edges):
    return {'m': edges.src['t']}

def to_real(t):
    r_list = []

    for tt in t:
        tt = tt[:real_list(tt)]
        r_list.append(tt)

    return r_list

@th.no_grad()
def mask_init(g,t):
    mask_list = []
    for i in range(g.__len__()):
        ind = t[i]

        if ind == [0] * len(ind):
            ind = []

        mask_i = th.LongTensor([[0] * len(ind), ind]).to('cuda:0')
        mask_v = th.ones(len(ind)).to('cuda:0')
        mask = th.sparse.FloatTensor(mask_i, mask_v, th.Size([1, g.__len__()])).to_dense().to('cuda:0')
        mask_list.append(mask)

    mask_batch = th.stack(tuple(mask_list)).squeeze()

    return mask_batch

def aggregate_init(g):
    for i in range(g.number_of_nodes()):
        with th.no_grad():
            g.nodes[i].data['id'] = th.tensor([i]).to('cuda:0')
    
    with th.no_grad():
        g.ndata['t'] = (th.ones([g.number_of_nodes(), ph], dtype=th.int64) * -1).to('cuda:0')
        g.ndata['tt'] = (th.ones([g.number_of_nodes(), ph], dtype=th.int64) * -1).to('cuda:0')
    
    g.register_message_func(id_to_m)
    g.register_reduce_func(m_to_t)
    
    g.send(g.edges())
    g.recv(g.nodes())
    
    g.register_message_func(t_to_m)
    g.register_reduce_func(m_to_tt)
    
    g.send(g.edges())
    g.recv(g.nodes())

    t = to_real(g.ndata['t'].tolist())
    tt = to_real(g.ndata['t'].tolist())

    mask_t = mask_init(g, t)
    mask_tt = mask_init(g, tt)

    return t, tt, mask_t, mask_tt

class GNNModule(nn.Module):
    def __init__(self, in_feats, out_feats, radius, dev):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.radius = radius

        self.dev = dev

        new_linear = lambda: nn.Linear(in_feats, out_feats)
        new_linear_list = lambda: nn.ModuleList([new_linear() for i in range(radius)])

        self.theta_x, self.theta_deg, self.theta_y = \
            new_linear(), new_linear(), new_linear()
        self.theta_list = new_linear_list()

        self.gamma_y, self.gamma_deg, self.gamma_x = \
            new_linear(), new_linear(), new_linear()
        self.gamma_list = new_linear_list()

        self.bn_x = nn.BatchNorm1d(out_feats)
        self.bn_y = nn.BatchNorm1d(out_feats)

    def forward(self, g, lg, x, y, deg_g, deg_lg, pm_pd, g_t, g_tt, lg_t, lg_tt, mask_g_t, mask_g_tt, mask_lg_t, mask_lg_tt):
        pmpd_x = F.embedding(pm_pd, x)
        
        g.set_n_repr({'x': x})

        x_list = []

        if self.in_feats == 1:
            x_list.append(th.mm(mask_g_t, g.ndata['x']).squeeze(1).reshape(-1,1)) 
            x_list.append(th.mm(mask_g_tt, g.ndata['x']).squeeze(1).reshape(-1,1))
        else:
            x_list.append(th.mm(mask_g_t, g.ndata['x']).squeeze(1))
            x_list.append(th.mm(mask_g_tt, g.ndata['x']).squeeze(1))

        sum_x = sum(theta(z) for theta, z in zip(self.theta_list, x_list))

        g.set_e_repr({'y' : y})
        g.update_all(fn.copy_edge(edge='y', out='m'), fn.sum('m', 'pmpd_y'))
        pmpd_y = g.pop_n_repr('pmpd_y')

        x = self.theta_x(x) + self.theta_deg(deg_g * x) + sum_x + self.theta_y(pmpd_y)
        n = self.out_feats // 2
        x = th.cat([x[:, :n], F.relu(x[:, n:])], 1)
        x = self.bn_x(x)

        y_list = []
        
        lg.set_n_repr({'y': y})

        if self.in_feats == 1:
            y_list.append(th.mm(mask_lg_t, lg.ndata['y']).squeeze(1).reshape(-1,1))
            y_list.append(th.mm(mask_lg_tt, lg.ndata['y']).squeeze(1).reshape(-1,1))
        else:
            y_list.append(th.mm(mask_lg_t, lg.ndata['y']).squeeze(1))
            y_list.append(th.mm(mask_lg_tt, lg.ndata['y']).squeeze(1))
        
        sum_y = sum(gamma(z) for gamma, z in zip(self.gamma_list, y_list))

        y = self.gamma_y(y) + self.gamma_deg(deg_lg * y) + sum_y + self.gamma_x(pmpd_x)
        y = th.cat([y[:, :n], F.relu(y[:, n:])], 1)
        y = self.bn_y(y)

        return x, y

class GNN(nn.Module):
    def __init__(self, feats, radius, n_classes, dev):
        """
        Parameters
        ----------
        g : networkx.DiGraph
        """
        super(GNN, self).__init__()
        self.linear = nn.Linear(feats[-1], n_classes)
        self.module_list = nn.ModuleList([GNNModule(m, n, radius, dev)
                                          for m, n in zip(feats[:-1], feats[1:])])

    def forward(self, g, lg, deg_g, deg_lg, pm_pd, g_t, g_tt, lg_t, lg_tt, mask_g_t, mask_g_tt, mask_lg_t, mask_lg_tt):
        x, y = deg_g, deg_lg
        for module in self.module_list:
            x, y = module(g, lg, x, y, deg_g, deg_lg, pm_pd, g_t, g_tt, lg_t, lg_tt, mask_g_t, mask_g_tt, mask_lg_t, mask_lg_tt)
        return self.linear(x)

