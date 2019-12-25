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

msg_h = fn.copy_src(src='h', out='m_h')
msg_w = fn.copy_src(src='w', out='m_w')


def reduce(nodes):
    accum = torch.cat((torch.sum(nodes.mailbox['m'], 1), torch.max(nodes.mailbox['m'], 1)[0]), dim=1)
    return {'hm': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(3*in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(torch.cat((node.data['h'], node.data['hm']), dim=1))
        h = self.activation(h)
        return {'h': h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        g.ndata.pop('hm')
        return g.ndata.pop('h')

g = nx.fast_gnp_random_graph(20, 0.5)
g = dgl.DGLGraph(g)
# gcn = GCN(2, 2, F.relu)
# 2-d coordinates 'x'
g.ndata['x'] = torch.rand((g.number_of_nodes(), 2))
g.ndata['label'] = torch.ones((g.number_of_nodes(), 1))
g.edata['e_type'] = torch.zeros((g.number_of_edges(), 1))
g.edata['d'] = torch.zeros((g.number_of_edges(), 1))
g.edata['w'] = torch.ones((g.number_of_edges(), 1))
# store the dist-matrix
a = 1
for i in range(g.number_of_edges()):
    v1 = g.edges()[0][i]
    v2 = g.edges()[1][i]
    x1 = g.nodes[v1].data['x']
    x2 = g.nodes[v2].data['x']
    d = torch.cdist(x1, x2)
    g.edges[v1, v2].data['d'] = d
    g.edges[v1, v2].data['w'] = 1 / (1 + torch.exp(a * d))



g.ndata['h'] = torch.rand((g.number_of_nodes(), 3))

# g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_sum'))

def udf_u_mul_e(edges):
   return {'m1': edges.src['h'] * edges.data['w'], 'm2': edges.data['w']}

l0 = nn.Linear(1, 3)
l1 = nn.Linear(3, 3)
l2 = nn.Linear(2, 3)

act = F.relu
def reduce(nodes):
    l = nodes.data['label']
    h = torch.sum(nodes.mailbox['m1'], 1) / torch.sum(nodes.mailbox['m2'], 1)
    x = nodes.data['x']

    accum = act(l0(l)+l1(h)+l2(x)+l3())
    return {'h': accum}

for i in range(100):
    # g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
    g.update_all(udf_u_mul_e, reduce)
g.ndata['h']

# g.apply_edges(fn.u_mul_v('h', 'h', 'w_new'))



h = g.ndata['x']
for i in range(500):
    h = gcn(g, h)
h

class ACNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ACNet, self).__init__()
        
        self.policy = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])

    def forward(self, g):
        h = g.ndata['x']
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        mN = dgl.mean_nodes(g, 'h')
        PI = self.policy(g.ndata['h'])
        V = self.value(mN)
        g.ndata.pop('h')
        return PI, V
