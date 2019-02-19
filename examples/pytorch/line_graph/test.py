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
import gnn
from gnn import aggregate_init

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
parser.add_argument('--gpu', type=int, help='GPU index', default=-1)
parser.add_argument('--n-communities', type=int, help='Number of communities', default=5)
parser.add_argument('--n-features', type=int, help='Number of features', default=8)
parser.add_argument('--n-graphs', type=int, help='Number of graphs', default=100)
parser.add_argument('--p', type=float, help='intra-community probability', default=0)
parser.add_argument('--q', type=float, help='inter-comminity probability', default=18)
parser.add_argument('--n-layers', type=int, help='Number of layers', default=30)
parser.add_argument('--n-nodes', type=int, help='Number of nodes', default=400)
parser.add_argument('--radius', type=int, help='Radius', default=2)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--model-path', type=str, default='model.pkl')
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

##feats = [1] + [args.n_features] * args.n_layers + [K]
feats = [1] + [args.n_features] * args.n_layers
model = gnn.GNN(feats, args.radius, K).to(dev)
with th.no_grad():
    model.load_state_dict(th.load(args.model_path))
#model.eval()

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

@from_np
def inference(g, lg, deg_g, deg_lg, pm_pd):
    deg_g = deg_g.to(dev)
    deg_lg = deg_lg.to(dev)
    pm_pd = pm_pd.to(dev)

    z = model(g, lg, deg_g, deg_lg, pm_pd)

    return z

def test():
    overlap_list = []

    N = 1

    for i in range(args.n_graphs):
        g, lg, deg_g, deg_lg, pm_pd = sbm.SBM(1, args.n_nodes, K, p, q).__getitem__(0)
        aggregate_init(g)
        aggregate_init(lg)
        z = inference(g, lg, deg_g, deg_lg, pm_pd)
        overlap = compute_overlap(th.chunk(z, N, 0))
        ##print('[test %d] overlap %.3f' % (i,  overlap))
        overlap_list.append(overlap)

    return overlap_list

overlap_list = test()

print('Node: %d | Average: %.3f | Variance: %.3f' % (args.n_nodes, np.mean(overlap_list), np.std(overlap_list)))
