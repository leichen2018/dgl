"""
Supervised Community Detection with Hierarchical Graph Neural Networks
https://arxiv.org/abs/1705.08415

Author's implementation: https://github.com/joanbruna/GNN_community
"""

from __future__ import division
import time
from datetime import datetime as dt

import argparse
from itertools import permutations

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dgl.data import SBMMixture
import sbm
import gnn_2

import sys

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
parser.add_argument('--gpu', type=int, help='GPU index', default=-1)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.004)
parser.add_argument('--n-communities', type=int, help='Number of communities', default=5)
parser.add_argument('--n-epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--n-features', type=int, help='Number of features', default=8)
parser.add_argument('--n-graphs', type=int, help='Number of graphs', default=6000)
parser.add_argument('--p', type=float, help='intra-community probability', default=0)
parser.add_argument('--q', type=float, help='inter-comminity probability', default=18)
parser.add_argument('--n-layers', type=int, help='Number of layers', default=30)
parser.add_argument('--n-nodes', type=int, help='Number of nodes', default=400)
parser.add_argument('--optim', type=str, help='Optimizer', default='Adamax')
parser.add_argument('--radius', type=int, help='Radius', default=2)
parser.add_argument('--clip_grad_norm', type=float, default=40.0)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--save-path', type=str, default='model')
parser.add_argument('--interval', type=int, help='loss intervel', default=50)
parser.add_argument('--model', type=int, default=0)
args = parser.parse_args()

dev = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)
K = args.n_communities
p = args.p
q = args.q

##training_dataset = sbm.SBM(args.n_graphs, args.n_nodes, K, p, q)
##training_dataset = SBMMixture(args.n_graphs, args.n_nodes, K)
##training_loader = DataLoader(training_dataset, args.batch_size, collate_fn=training_dataset.collate_fn, drop_last=True)

ones = th.ones(args.n_nodes // K)
y_list = [th.cat([x * ones for x in p]).long().to(dev) for p in permutations(range(K))]

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

feats = [1] + [args.n_features] * args.n_layers + [K]

model = gnn_2.GNN(feats, args.radius, K, dev).to(dev)

optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

def compute_overlap(z_list):
    ybar_list = [th.max(z, 1)[1] for z in z_list]
    overlap_list = []
    for y_bar in ybar_list:
        accuracy = max(th.sum(y_bar == y).item() for y in y_list) / args.n_nodes
        overlap = (accuracy - 1 / K) / (1 - 1 / K)
        overlap_list.append(overlap)
    return sum(overlap_list) / len(overlap_list)

def from_np(f, *args):
    def wrap(*args):
        new = [th.from_numpy(x) if isinstance(x, np.ndarray) else x for x in args]
        return f(*new)
    return wrap

@th.no_grad()
def succ_random(g):
    succ_list = []
    
    for i in range(g.__len__()):
        if g.successors(i).size()[0] > 0:
            succ_list.append(g.successors(i)[th.randint(0, g.successors(i).size()[0], (1,))])
        #    succ_list.append(g.successors(i)[-1])
    
    succ = th.cat(tuple(succ_list), dim=0)
    #succ = th.stack(tuple(succ_list))

    return succ

@th.no_grad()
def adj2_gen(a1, succ):
    mask_i = th.sparse.FloatTensor(th.stack((th.arange(succ.size()[0]).to('cuda:0'), succ)), th.ones(succ.size()).to('cuda:0'), \
        a1.size()).to('cuda:0')
    adj2 = th.sparse.mm(mask_i, a1)
    
    return adj2

@from_np
def step(i, j, deg_g, g_a1, g_a2):
    """ One step of training. """
    deg_g = deg_g.to(dev)
    t0 = time.time()
    z = model(deg_g, g_a1, g_a2)

    t_forward = time.time() - t0

    z_list = th.chunk(z, args.batch_size, 0)
    loss = sum(min(F.cross_entropy(z, y) for y in y_list) for z in z_list) / args.batch_size
    overlap = compute_overlap(z_list)

    optimizer.zero_grad()
    #model.zero_grad()
    t0 = time.time()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    t_backward = time.time() - t0
    optimizer.step()

    return loss, overlap, t_forward, t_backward

n_iterations = args.n_graphs // args.batch_size
for i in range(args.n_epochs):
    total_loss, total_overlap, s_forward, s_backward, s_buildgraph = 0, 0, 0, 0, 0
    interval_loss, interval_overlap = 0, 0
##    for j, [g, lg, deg_g, deg_lg, pm_pd] in enumerate(training_loader):
    for j in range(args.n_graphs):
        with th.no_grad():	
            g, lg, deg_g, deg_lg, g_adj, lg_adj = sbm.SBM(1, args.n_nodes, K, p, q).__getitem__(0)

            g_a1 = g_adj.to_dense()
            g_a1_a1 = th.sparse.mm(g_adj, g_adj.to_dense())
            g_a2 = th.where(g_a1_a1>0, th.ones(g_a1_a1.size()).to('cuda:0'), g_a1_a1)
            
            #g_a2_deg = th.diag(1/th.sum(g_a2, 0)).to('cuda:0')

            #g_a2 = th.mm(g_a2_deg, g_a2)
            
            #g_a2 = adj2_gen(g_a1, succ_random(g).to('cuda:0'))
            #g_a2 = th.mm(th.diag(1/th.sum(g_a2, 0)).to('cuda:0'), g_a2)
            #g_a2 = g_a2 - th.diag(th.sum(g_a2, 0)).to('cuda:0')

            #if j<1500:
            #    g_a2 = th.zeros(g_a2.size()).to('cuda:0')

            #print(g_a2)
            
        loss, overlap, t_forward, t_backward = step(i, j, deg_g, g_a1, g_a2)

        total_loss += loss.item()
        total_overlap += overlap
        s_forward += t_forward
        s_backward += t_backward

        interval_loss += loss.item()
        interval_overlap += overlap

        epoch = '0' * (len(str(args.n_epochs)) - len(str(i)))
        iteration = '0' * (len(str(n_iterations)) - len(str(j)))
        if args.verbose:
            print(dt.now(), '[epoch %s%d iteration %s%d]loss %.3f | overlap %.3f'
                  % (epoch, i, iteration, j, loss, overlap))
        
        if (j+1) % args.interval == 0:
            if j > 0:
                print('=================== interval %d | loss %0.3f | overlap %.3f ===================' 
                % (j, interval_loss/args.interval, interval_overlap/args.interval))

            interval_loss = 0
            interval_overlap = 0

    epoch = '0' * (len(str(args.n_epochs)) - len(str(i)))
    loss = total_loss / (j + 1)
    overlap = total_overlap / (j + 1)
    t_forward = s_forward / (j + 1)
    t_backward = s_backward / (j + 1)
    t_buildgraph = s_buildgraph / (j + 1)
    print('[epoch %s%d]loss %.3f | overlap %.3f | forward time %.3fs | backward time %.3fs | buildgraph time %.3fs'
          % (epoch, i, loss, overlap, t_forward, t_backward, t_buildgraph))

    #overlap_list = test()
    #overlap_str = ' - '.join(['%.3f' % overlap for overlap in overlap_list])
    #print('[epoch %s%d]overlap: %s' % (epoch, i, overlap_str))

th.save({
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict() 
}, 'checkpoints/' + args.save_path + '.pkl')