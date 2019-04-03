"""Dataset for stochastic block model."""
import math
import os
import pickle
import random

import numpy as np
import numpy.random as npr
import scipy as sp
import networkx as nx

import torch

from dgl import backend as F

from dgl.batched_graph import batch
from dgl.graph import DGLGraph
from dgl.utils import Index

import dgl.data as data

class SBM:
    """ Symmetric Stochastic Block Model

    Parameters
    ----------
    n_graphs : int
        Number of graphs.
    n_nodes : int
        Number of nodes.
    n_communities : int
        Number of communities.
    p : float
        Probability for intra-community edge.
    q : float
        Probability for inter-community edge.
    rng : numpy.random.RandomState, optional
        Random number generator.
    """
    def __init__(self, n_graphs, n_nodes, n_communities, p, q, rng=None):
        self._n_nodes = n_nodes
        assert n_nodes % n_communities == 0
        block_size = n_nodes // n_communities
        p /= n_nodes
        q /= n_nodes
        self._gs = [DGLGraph() for i in range(n_graphs)]
        for g in self._gs:
            g_nx = nx.planted_partition_graph(n_communities, block_size, p, q)
            g.from_networkx(g_nx)
        self._lgs = [g.line_graph(backtracking=False) for g in self._gs]
        in_degrees = lambda g: g.in_degrees(
                Index(np.arange(0, g.number_of_nodes()))).unsqueeze(1).float()
        out_degrees = lambda g: g.out_degrees(
                Index(np.arange(0, g.number_of_nodes()))).unsqueeze(1).float()
        self._g_degs = [in_degrees(g) for g in self._gs]
        self._lg_degs = [out_degrees(lg) for lg in self._lgs]

        self._g_adj = [g.adjacency_matrix().to('cuda:0') for g in self._gs]
        self._lg_adj = [lg.adjacency_matrix().to('cuda:0') for lg in self._lgs]

    def __len__(self):
        return len(self._gs)

    def __getitem__(self, idx):
        return self._gs[idx], self._lgs[idx], \
                self._g_degs[idx], self._lg_degs[idx], self._g_adj[idx], self._lg_adj[idx]

    def collate_fn(self, x):
        g, lg, deg_g, deg_lg, pm_pd = zip(*x)
        g_batch = batch(g)
        lg_batch = batch(lg)
        degg_batch = np.concatenate(deg_g, axis=0)
        deglg_batch = np.concatenate(deg_lg, axis=0)
        return g_batch, lg_batch, degg_batch, deglg_batch
