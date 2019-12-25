# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:28:50 2019

@author: orrivlin
"""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, in_dim, n_hidden, num_grp, activation):
        super(GNN, self).__init__()
        self.activation = activation
        self.embedding_x = nn.Linear(in_dim, n_hidden)
        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)

        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden + num_grp, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, state):
        x = state.X
        label = state.label
        label_onehot = torch.nn.functional.one_hot(label, state.num_grp).float()
        x_embedding = self.embedding_x(x)
        x1 = self.activation(self.agg_1(torch.cat([x_embedding, label_onehot], axis=1)))
        x2 = self.activation(self.agg_2(x1))
        return x2

    # def forward(self, state, action):
    #     i = action[0]
    #     j = action[1]
    #     x = torch.cat([state.X, state.X[i].unsqueeze(0), state.X[j].unsqueeze(0)], axis=0)
    #     label = torch.cat([state.label, state.label[i].unsqueeze(0), state.label[j].unsqueeze(0)], axis=0)
    #     label_onehot = torch.nn.functional.one_hot(label, state.num_grp)
    #     x_embedding = self.embedding_x(x)
    #     x1 = self.activation(self.agg_1(torch.cat([x_embedding, label_onehot])))
    #     x2 = self.activation(self.agg_1(x1))
    #     return x2.sum(axis=0)


class ACNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_grp, n):
        super(ACNet, self).__init__()
        
        self.policy = nn.Linear(hidden_dim*n, n*n)
        self.value = nn.Linear(hidden_dim*n, 1)
        self.valueM = nn.Linear(hidden_dim*n, n*n)
        self.layers = nn.ModuleList([
            GNN(in_dim, hidden_dim, num_grp, F.relu)])

    def forward(self, state):
        for conv in self.layers:
            h = conv(state)
        h = h.view(-1)
        PI = self.policy(h)
        V = self.value(h)
        VM = self.valueM(h)
        return PI, V, VM


