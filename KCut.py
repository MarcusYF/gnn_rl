# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:01:56 2019

@author: Or
"""

import dgl
import torch
import networkx as nx
import numpy as np
import itertools
from copy import deepcopy as dc


def init_state(N, k):
    num_grp = N//k
    X = torch.rand((N, 2))
    label = torch.tensor(range(num_grp)).unsqueeze(0).expand(k, num_grp).t().flatten()
    # label1 = torch.nn.functional.one_hot(label, num_grp)
    return X, label#, label1

class KCut:

    def __init__(self, N, k):
        self.N = N
        self.k = k
        self.num_grp = N//k
        self.X, self.label = init_state(self.N, self.k)
        self.S = self.calc_S()

    def get_graph_dims(self):
        return 2

    def calc_S(self):
        S = 0
        for i in range(self.num_grp):
            block_i = self.X[self.label == i]
            for j in range(self.k):
                block_ij = block_i - block_i[j]
                S += torch.sum(torch.sqrt(torch.diag(torch.mm(block_ij, block_ij.t()))))
        return S/2

    def reset(self):
        self.X, self.label = init_state(self.N, self.k)
        return self.X, self.label

    def get_legal_actions(self):
        return list(itertools.product(range(self.N), range(self.N)))

    def step(self, action):

        old_0 = self.X[action[0]] - self.X[self.label == self.label[action[0]]]
        old_1 = self.X[action[1]] - self.X[self.label == self.label[action[1]]]

        # swap two nodes
        tmp = dc(self.label[action[0]])
        self.label[action[0]] = self.label[action[1]]
        self.label[action[1]] = tmp

        new_0 = self.X[action[0]] - self.X[self.label == self.label[action[0]]]
        new_1 = self.X[action[1]] - self.X[self.label == self.label[action[1]]]

        reward = torch.sqrt(torch.sum(torch.pow(torch.cat([old_0, old_1]), 2), axis=1)).sum() - torch.sqrt(torch.sum(torch.pow(torch.cat([new_0, new_1]), 2), axis=1)).sum()

        self.S -= reward

        return self.label, reward

p = KCut(15, 3)