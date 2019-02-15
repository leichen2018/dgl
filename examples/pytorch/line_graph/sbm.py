"""Dataset for stochastic block model."""
import math
import os
import pickle
import random

import numpy as np
import numpy.random as npr
import scipy as sp
import networkx as nx

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
        self._gs = [DGLGraph() for i in range(n_graphs)]
        adjs = [data.sbm.sbm(n_communities, block_size, p, q) for i in range(n_graphs)]
        for g, adj in zip(self._gs, adjs):
            g.from_scipy_sparse_matrix(adj)
        self._lgs = [g.line_graph(backtracking=False) for g in self._gs]
        in_degrees = lambda g: g.in_degrees(
                Index(np.arange(0, g.number_of_nodes()))).unsqueeze(1).float()
        self._g_degs = [in_degrees(g) for g in self._gs]
        self._lg_degs = [in_degrees(lg) for lg in self._lgs]
        self._pm_pds = list(zip(*[g.edges() for g in self._gs]))[0]

    def __len__(self):
        return len(self._gs)

    def __getitem__(self, idx):
        return self._gs[idx], self._lgs[idx], \
                self._g_degs[idx], self._lg_degs[idx], self._pm_pds[idx]

    def collate_fn(self, x):
        g, lg, deg_g, deg_lg, pm_pd = zip(*x)
        g_batch = batch(g)
        lg_batch = batch(lg)
        degg_batch = np.concatenate(deg_g, axis=0)
        deglg_batch = np.concatenate(deg_lg, axis=0)
        pm_pd_batch = np.concatenate([x + i * self._n_nodes for i, x in enumerate(pm_pd)], axis=0)
        return g_batch, lg_batch, degg_batch, deglg_batch, pm_pd_batch