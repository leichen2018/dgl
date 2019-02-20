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

def pp(l):
    for i in range(len(l)):
        print(l[i])

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

def printnode(g):
    for i in range(g.number_of_nodes()):
        print(g.nodes[i].data['id'])
        print(g.nodes[i].data['z'])
        print(g.nodes[i].data['t'])
    #print(node.data['zz'])

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

    t = g.ndata['t'].tolist()
    tt = g.ndata['t'].tolist()

    return to_real(t), to_real(tt)

def t_to_feature(g, t, in_feats):

    def nested_mes(nodes):
        zz_list = []
        for i in range(nodes.__len__()):
            ind = t[i]

            if len(ind) == 0 or ind == [0]*ph:
                ind = [i]

            if in_feats == 1:
                zz_list.append(th.sum(g.nodes[tuple(ind)].data['z'], dim=0))
            else:
                zz_list.append(th.sum(g.nodes[tuple(ind)].data['z'], dim=0).unsqueeze(0))

        return {'zz': th.stack(tuple(zz_list)).squeeze()}

    return nested_mes

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
    
    '''
    def t_to_feature(self, g, t):
        #print(g.ndata['z'])
        for i in range(g.number_of_nodes()):
            ind = t[i]
           
            if len(ind)==0 or (len(ind) == 1 and ind[0] == 0):
                 ind = [i]
            
            if self.in_feats == 1:
                g.nodes[i].data['zz'] = th.sum(g.nodes[tuple(ind)].data['z'], dim=0)
            else:
                g.nodes[i].data['zz'] = th.sum(g.nodes[tuple(ind)].data['z'], dim=0).unsqueeze(0)
        
    def t_to_feature(self, g):
        for i in range(g.number_of_nodes()):
            ind = g.nodes[i].data['t'][0].tolist()
            
            if ind == [0]*ph
            
    def tt_to_feature(self, g, tt):
        #print(g.ndata['z'])
        for i in range(g.number_of_nodes()):
            ind = tt[i]
            
            if len(ind)==0 or (len(ind) == 1 and ind[0] == 0):
                ind = [i]

            if self.in_feats == 1:
                g.nodes[i].data['zz'] = th.sum(g.nodes[tuple(ind)].data['z'], dim=0)
            else:
                g.nodes[i].data['zz'] = th.sum(g.nodes[tuple(ind)].data['z'], dim=0).unsqueeze(0)
            #print('============ t2 =========== %.9fs' % (time.time()-t0))
    '''

    def aggregate(self, g, z, t, tt):
        ###g.register_message_func(id_to_m)
        
        ###g.register_reduce_func(m_to_t)
        
 
        z_list = []
        g.set_n_repr({'z' : z})
        
        ###self.aggregate_init(g)
        #g.update_all(fn.copy_src(src='id', out='m'), fn.sum(msg='m', out='z'))
        ###g.send(g.edges())
        
        ###g.recv(g.nodes())
        
        ###g.register_message_func(t_to_m)
        
        #self.t_to_feature(g, t)
        g.apply_nodes(func=t_to_feature(g, t, self.in_feats), v=g.nodes())
        
        if self.in_feats == 1:
            z = g.ndata.pop('zz').reshape(-1,1)
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
        #self.tt_to_feature(g, tt)
        g.apply_nodes(func=t_to_feature(g, tt, self.in_feats), v=g.nodes())
            
        if self.in_feats == 1:
            z = g.ndata.pop('zz').reshape(-1,1)
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

    def forward(self, g, lg, x, y, deg_g, deg_lg, pm_pd, g_t, g_tt, lg_t, lg_tt):
        pmpd_x = F.embedding(pm_pd, x)
        
        t0 = time.time()
        ##print(pmpd_x)
        
        a = self.aggregate(g,x, g_t, g_tt)
        
        ##print(a)
        
        sum_x = sum(theta(z) for theta, z in zip(self.theta_list, a))

        g.set_e_repr({'y' : y})
        g.update_all(fn.copy_edge(edge='y', out='m'), fn.sum('m', 'pmpd_y'))
        pmpd_y = g.pop_n_repr('pmpd_y')

        x = self.theta_x(x) + self.theta_deg(deg_g * x) + sum_x + self.theta_y(pmpd_y)
        n = self.out_feats // 2
        x = th.cat([x[:, :n], F.relu(x[:, n:])], 1)
        x = self.bn_x(x)

        sum_y = sum(gamma(z) for gamma, z in zip(self.gamma_list, self.aggregate(lg, y, lg_t, lg_tt)))

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

    def forward(self, g, lg, deg_g, deg_lg, pm_pd, g_t, g_tt, lg_t, lg_tt):
        x, y = deg_g, deg_lg
        for module in self.module_list:
            x, y = module(g, lg, x, y, deg_g, deg_lg, pm_pd, g_t, g_tt, lg_t, lg_tt)
        return self.linear(x)

