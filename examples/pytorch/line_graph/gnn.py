import copy
import itertools
import dgl
import dgl.function as fn
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ph = 1000

def pp(l):
    for i in range(len(l)):
        print(l[i])

def printnode(g):
    for i in range(g.number_of_nodes()):
        print(g.nodes[i].data['id'])
        print(g.nodes[i].data['z'])
        print(g.nodes[i].data['t'])
    #print(node.data['zz'])

def id_to_m(edges):
    return {'m': edges.src['id']}

def m_to_t(nodes):
    #print(nodes.__len__())
    #print(nodes.nodes())
    new_t_list = []
    for i in range(nodes.__len__()):
        cur = nodes.data['t'][i].tolist()
        
        if not isinstance(cur, list):
            cur = [cur]
            
        new_t = cur[:cur.index(-1)]
        
        for m in nodes.mailbox['m'][i]:
            m = m.tolist()
            if isinstance(m, list):
                if -1 in m:
                    new_t = new_t + m[:m.index(-1)]
                else:
                    new_t = new_t + m
            else:
                new_t.append(m)
        
        new_t = list(set(new_t))
        
        if len(new_t) < ph:
            new_t = new_t + [-1] * (ph-len(new_t))
        
        new_t_list.append(new_t)
        
    #print(th.tensor(new_t_list))
    return {'t': th.tensor(new_t_list)}

def m_to_tt(nodes):
    #print(nodes.__len__())
    #print(nodes.nodes())
    new_t_list = []
    for i in range(nodes.__len__()):
        cur = nodes.data['t'][i].tolist()
        
        if not isinstance(cur, list):
            cur = [cur]
            
        new_t = cur[:cur.index(-1)]
        
        for m in nodes.mailbox['m'][i]:
            m = m.tolist()
            if isinstance(m, list):
                if -1 in m:
                    new_t = new_t + m[:m.index(-1)]
                else:
                    new_t = new_t + m
            else:
                new_t.append(m)
        
        new_t = list(set(new_t))
        
        if len(new_t) < ph:
            new_t = new_t + [-1] * (ph-len(new_t))
        
        new_t_list.append(new_t)
        
    #print(th.tensor(new_t_list))
    return {'tt': th.tensor(new_t_list)}

def t_to_m(edges):
    return {'m': edges.src['t']}

def aggregate_init(g):
    for i in range(g.number_of_nodes()):
        g.nodes[i].data['id'] = th.tensor([i])
    
    g.ndata['t'] = th.ones([g.number_of_nodes(), ph], dtype=th.int64) * -1
    g.ndata['tt'] = th.ones([g.number_of_nodes(), ph], dtype=th.int64) * -1
    
    g.register_message_func(id_to_m)
    g.register_reduce_func(m_to_t)
    
    g.send(g.edges())
    g.recv(g.nodes())
    
    g.register_message_func(t_to_m)
    g.register_reduce_func(m_to_tt)
    
    g.send(g.edges())
    g.recv(g.nodes())

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
    
    def t_to_feature(self, g):
        #print(g.ndata['z'])
        for i in range(g.number_of_nodes()):
            ind = g.nodes[i].data['t'][0].tolist()
            
            if ind == [0] * ph or ind[0] == -1:
                ind = [i]
            elif -1 in ind:
                ind = ind[:ind.index(-1)]
                
            ind = th.tensor(ind)
            #print("================  t_to_feature ============")
            #print(g.ndata['z'])
            #print(ind)
            #print((g.nodes[ind].data['z']))
            #print(th.sum(g.nodes[ind].data['z'], dim=0))
            #print(th.tensor([th.sum(g.nodes[ind].data['z'], dim=0).tolist()]))
            if self.in_feats == 1:
                g.nodes[i].data['zz'] = th.sum(g.nodes[ind].data['z'], dim=0)
            else:
                g.nodes[i].data['zz'] = th.tensor([th.sum(g.nodes[ind].data['z'], dim=0).tolist()])
        
            
    def tt_to_feature(self, g):
        #print(g.ndata['z'])
        for i in range(g.number_of_nodes()):
            ind = g.nodes[i].data['tt'][0].tolist()
            
            if ind == [0] * ph or ind[0] == -1:
                ind = [i]
            elif -1 in ind:
                ind = ind[:ind.index(-1)]
                
            ind = th.tensor(ind)
            #print("================  t_to_feature ============")
            #print(g.ndata['z'])
            #print(ind)
            #print((g.nodes[ind].data['z']))
            #print(th.sum(g.nodes[ind].data['z'], dim=0))
            #print(th.tensor([th.sum(g.nodes[ind].data['z'], dim=0).tolist()]))
            if self.in_feats == 1:
                g.nodes[i].data['zz'] = th.sum(g.nodes[ind].data['z'], dim=0)
            else:
                g.nodes[i].data['zz'] = th.tensor([th.sum(g.nodes[ind].data['z'], dim=0).tolist()])

    def aggregate(self, g, z):
        ###g.register_message_func(id_to_m)
        
        ###g.register_reduce_func(m_to_t)
        
        z_list = []
        g.set_n_repr({'z' : z})
        
        ###self.aggregate_init(g)
        #g.update_all(fn.copy_src(src='id', out='m'), fn.sum(msg='m', out='z'))
        ###g.send(g.edges())
        
        ###g.recv(g.nodes())
        
        ###g.register_message_func(t_to_m)
        
        self.t_to_feature(g)
        
        if self.in_feats == 1:
            z = th.tensor([[b] for b in g.ndata.pop('zz').tolist()])
        else:
            z = g.ndata.pop('zz')
        
        z_list.append(z)
        '''
        for i in range(self.radius-1):
           for j in range(2 ** i):
                g.send(g.edges())
                g.recv(g.nodes())
                #g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
        '''    
        self.tt_to_feature(g)
            
        if self.in_feats == 1:
            z = th.tensor([[b] for b in g.ndata.pop('zz').tolist()])
        else:
            z = g.ndata.pop('zz')
            
            #print('============== z ==============')
            #print(z)
            
        z_list.append(z)
        
        #print('==============. g  =========')
        #printnode(g)
        #print('==============. zlist  =========')
        #print(z_list)
        return z_list

    def forward(self, g, lg, x, y, deg_g, deg_lg, pm_pd):
        pmpd_x = F.embedding(pm_pd, x)
        
        ##print(pmpd_x)
        
        a = self.aggregate(g,x)
        
        ##print(a)
        
        sum_x = sum(theta(z.to(self.dev)) for theta, z in zip(self.theta_list, a))

        g.set_e_repr({'y' : y})
        g.update_all(fn.copy_edge(edge='y', out='m'), fn.sum('m', 'pmpd_y'))
        pmpd_y = g.pop_n_repr('pmpd_y')

        x = self.theta_x(x) + self.theta_deg(deg_g * x) + sum_x + self.theta_y(pmpd_y)
        n = self.out_feats // 2
        x = th.cat([x[:, :n], F.relu(x[:, n:])], 1)
        x = self.bn_x(x)

        sum_y = sum(gamma(z.to(self.dev)) for gamma, z in zip(self.gamma_list, self.aggregate(lg, y)))

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

    def forward(self, g, lg, deg_g, deg_lg, pm_pd):
        x, y = deg_g, deg_lg
        for module in self.module_list:
            x, y = module(g, lg, x, y, deg_g, deg_lg, pm_pd)
        return self.linear(x)

