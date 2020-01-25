import math
from copy import deepcopy as dc
import itertools
import numpy as np
import dgl
from dgl.graph import DGLGraph
from dgl.batched_graph import BatchedDGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_G(k, m, adjacent_reserve, hidden_dim, random_sample_node=True, x=None, random_init_label=True, label=None, a=1):

    # init graph
    g = dgl.DGLGraph()
    g.add_nodes(k*m)

    # 2-d coordinates 'x'
    if random_sample_node:
        g.ndata['x'] = torch.rand((k * m, 2))
    else:
        g.ndata['x'] = x
        # g.ndata['x'] = torch.tensor(
        #     [[0, 0], [0, 1], [1, 0], [1, 1], [0.7, 0.8], [1.5, 1.5], [1.5, 3], [3, 3], [3, 1.5], [1.8, 1.7], [1.8, 0.3],
        #      [1.8, 0.8], [2.3, 0.8], [2.3, 0.3], [2.1, 0.5]])

    if random_init_label:
        label = torch.tensor(range(k)).unsqueeze(1).expand(k, m).flatten()
        label = label[torch.randperm(k * m)]
    else:
        label = torch.tensor(label)

    g.ndata['label'] = torch.nn.functional.one_hot(label, k).float()

    # store the dist-matrix
    for i in range(g.number_of_nodes()):
        v = g.nodes[i]
        v_x = v.data['x']
        v_label = v.data['label']

        # find the top-k nearest neighbors of v
        neighbor_dict = {}
        for j in range(g.number_of_nodes()):
            if j != i:
                u = g.nodes[j]
                u_x = u.data['x']
                d = torch.cdist(v_x, u_x)
                # form a complete graph
                g.add_edges(i, j)
                g.edges[i, j].data['d'] = d
                g.edges[i, j].data['w'] = 1 / (1 + torch.exp(a * d))
                g.edges[i, j].data['e_type'] = torch.zeros((1, 2))
                neighbor_dict[j] = d

        neighbor_dist = sorted(neighbor_dict.items(), key=lambda x: x[1], reverse=True)
        nearest_neighbors = [x[0] for x in neighbor_dist[k * m - 1 - adjacent_reserve:]]

        # add edges
        for j in range(g.number_of_nodes()):
            if j != i:
                u_label = g.nodes[j].data['label']
                if j in nearest_neighbors:
                    # add neighbors
                    g.edges[i, j].data['e_type'] += torch.tensor([[1, 0]])
                if torch.max(u_label-v_label).numpy() == 0:
                    # add group members
                    g.edges[i, j].data['e_type'] += torch.tensor([[0, 1]])

    # init node embedding h
    g.ndata['h'] = torch.zeros((g.number_of_nodes(), hidden_dim))

    G = {'g': g, 'k': k, 'm': m, 'adjacent_reserve': adjacent_reserve, 'hidden_dim': hidden_dim, 'a': a}
    return G


class KCut_DGL():

    def __init__(self, k, m, adjacent_reserve, hidden_dim, random_sample_node=True, x=None, random_init_label=True, label=None, a=1):
        self.g = generate_G(k, m, adjacent_reserve, hidden_dim
                            , random_sample_node=random_sample_node, x=x
                            , random_init_label=random_init_label, label=label
                            , a=a)['g']
        self.N = k * m
        self.k = k # num of clusters
        self.m = m # num of nodes in cluster
        self.adjacent_reserve = adjacent_reserve
        self.hidden_dim = hidden_dim
        self.random_sample_node = random_sample_node
        self.x = x
        self.random_init_label = random_init_label
        self.label = label
        self.a = a
        self.S = self.calc_S()

    def get_graph_dims(self):
        return 2

    def calc_S(self, g=None):
        if g is None:
            g = self.g
        S = 0
        for i in range(self.k):
            block_i = g.ndata['x'][(g.ndata['label'][:, i] > .5).nonzero().squeeze()]
            block_size = block_i.view(-1, 2).shape[0]
            if block_size < 2:
                continue
            for j in range(block_size):
                block_ij = block_i - block_i[j]
                S += torch.sum(torch.sqrt(torch.diag(torch.mm(block_ij, block_ij.t()))))
        return S/2

    def gen_batch_graph(self, batch_size):
        return [generate_G(self.k, self.m, self.adjacent_reserve, self.hidden_dim, random_sample_node=self.random_sample_node, x=self.x, random_init_label=self.random_init_label, label=self.label, a=self.a)['g'] for i in range(batch_size)]

    def reset(self, compute_S=True):
        self.g = generate_G(self.k, self.m, self.adjacent_reserve, self.hidden_dim, random_sample_node=self.random_sample_node, x=self.x, random_init_label=self.random_init_label, label=self.label, a=self.a)['g']
        if compute_S:
            self.S = self.calc_S()
        return self.g

    def reset_label(self, label, g=None, calc_S=True, rewire_edges=True):
        if g is None:
            g = self.g
        label = torch.tensor(label)
        g.ndata['label'] = torch.nn.functional.one_hot(label, self.k).float()

        # rewire edges
        if rewire_edges:
            for i, j in itertools.product(range(self.N), range(self.N)):
                if j != i:
                    u_label = g.nodes[j].data['label']
                    v_label = g.nodes[i].data['label']
                    if torch.max(u_label-v_label).numpy() < .5:
                        cluster_label = torch.tensor([[1.]])
                    else:
                        cluster_label = torch.tensor([[0.]])
                    # add group members
                    g.edges[i, j].data['e_type'] \
                        = torch.cat([g.edges[i, j].data['e_type'][:, 0].unsqueeze(0), cluster_label], dim=1)

        if g is None and calc_S:
            self.S = self.calc_S()
        return g

    def get_legal_actions(self, state=None, update=False, action_type='swap', action=None):

        if state is None:
            state = self.g

        if action_type=='flip':
            legal_actions = torch.nonzero(1 - state.ndata['label'])
        if action_type=='swap':
            mask = torch.mm(state.ndata['label'], state.ndata['label'].t())
            legal_actions = torch.triu(1 - mask).nonzero()

        return legal_actions

    def calc_swap_delta(self, i, j, i_label, j_label, state=None):

        if state is None:
            state = self.g

        return state.ndata['x'][i] - state.ndata['x'][(state.ndata['label'][:, i_label] > 0).nonzero().squeeze()], \
                state.ndata['x'][j] - state.ndata['x'][(state.ndata['label'][:, j_label] > 0).nonzero().squeeze()]

    def calc_flip_delta(self, i, i_label, target_label, state=None):

        if state is None:
            state = self.g

        return state.ndata['x'][i:i+1] - state.ndata['x'][(state.ndata['label'][:, i_label] > 0).nonzero().squeeze()], \
               state.ndata['x'][i:i+1] - state.ndata['x'][(state.ndata['label'][:, target_label] > 0).nonzero().squeeze()]

    def step(self, action, action_type='swap', state=None, rewire_edges=True):

        if state is None:
            state = self.g

        if action_type == 'swap':
            i, j = action
            i_label, j_label = [(state.ndata['label'][c] > 0).nonzero().item() for c in action]

            if i == j or i_label == j_label:
                return state, torch.tensor(.0)

            # rewire edges
            if rewire_edges:
                group_i = (state.ndata['label'][:, i_label] > .5).nonzero().squeeze()
                group_j = (state.ndata['label'][:, j_label] > .5).nonzero().squeeze()
                state.edges[i, group_i].data['e_type'] -= torch.tensor([[0, 1]]).repeat(self.m - 1, 1)
                state.edges[group_i, i].data['e_type'] -= torch.tensor([[0, 1]]).repeat(self.m - 1, 1)
                state.edges[i, group_j].data['e_type'] += torch.tensor([[0, 1]]).repeat(self.m, 1)
                state.edges[group_j, i].data['e_type'] += torch.tensor([[0, 1]]).repeat(self.m, 1)
                state.edges[j, group_j].data['e_type'] -= torch.tensor([[0, 1]]).repeat(self.m - 1, 1)
                state.edges[group_j, j].data['e_type'] -= torch.tensor([[0, 1]]).repeat(self.m - 1, 1)
                state.edges[j, group_i].data['e_type'] += torch.tensor([[0, 1]]).repeat(self.m, 1)
                state.edges[group_i, j].data['e_type'] += torch.tensor([[0, 1]]).repeat(self.m, 1)
                state.edges[j, i].data['e_type'] -= torch.tensor([[0, 2]])
                state.edges[i, j].data['e_type'] -= torch.tensor([[0, 2]])

            # compute reward
            old_0, old_1 = self.calc_swap_delta(i, j, i_label, j_label, state)

            # swap two nodes
            tmp = dc(state.nodes[i].data['label'])
            state.nodes[i].data['label'] = state.nodes[j].data['label']
            state.nodes[j].data['label'] = tmp

            new_0, new_1 = self.calc_swap_delta(i, j, j_label, i_label, state)
            reward = torch.sqrt(torch.sum(torch.pow(torch.cat([old_0, old_1]), 2), axis=1)).sum()\
                     - torch.sqrt(torch.sum(torch.pow(torch.cat([new_0, new_1]), 2), axis=1)).sum()

        elif action_type == 'flip':
            i = action[0]
            target_label = action[1]
            i_label = state.nodes[i].data['label'].argmax().item()

            if i_label == target_label:
                return state, torch.tensor(0.0)

            # rewire edges
            if rewire_edges:
                group_i = (state.ndata['label'][:, i_label] > .5).nonzero().squeeze()
                group_j = (state.ndata['label'][:, target_label] > .5).nonzero().squeeze()
                n1 = state.edges[i, group_i].data['e_type'].shape[0]
                n2 = state.edges[i, group_j].data['e_type'].shape[0]
                state.edges[i, group_i].data['e_type'] -= torch.tensor([[0, 1]]).repeat(n1, 1)
                state.edges[group_i, i].data['e_type'] -= torch.tensor([[0, 1]]).repeat(n1, 1)
                state.edges[i, group_j].data['e_type'] += torch.tensor([[0, 1]]).repeat(n2, 1)
                state.edges[group_j, i].data['e_type'] += torch.tensor([[0, 1]]).repeat(n2, 1)

            # compute reward
            old, new = self.calc_flip_delta(i, i_label, target_label, state)

            if old.shape[0] + new.shape[0] < .5:
                reward = torch.tensor(0)
            elif new.shape[0] < .5:
                reward = torch.sqrt(torch.sum(torch.pow(old, 2), axis=1)).sum()
            elif old.shape[0] < .5:
                reward = - torch.sqrt(torch.sum(torch.pow(new, 2), axis=1)).sum()
            else:
                reward = torch.sqrt(torch.sum(torch.pow(old, 2), axis=1)).sum() \
                        - torch.sqrt(torch.sum(torch.pow(new, 2), axis=1)).sum()

            # flip node
            state.nodes[i].data['label'] = torch.nn.functional.one_hot(torch.tensor([target_label]), self.k).float()

        if state is None:
            self.S -= reward

        return state, reward


def udf_u_mul_e(edges):
    # a= edges.data['d'] * edges.data['e_type'][:, 0].unsqueeze(1)
    # print(a.view(15,14,1))
    return {
        'm_n1_h': edges.src['h'] * edges.data['e_type'][:, 0].unsqueeze(1)
        , 'm_n1_v': edges.src['h'] * edges.data['w'] * edges.data['e_type'][:, 0].unsqueeze(1)
        , 'm_n1_w': edges.data['w'] * edges.data['e_type'][:, 0].unsqueeze(1)
        , 'm_n1_d': edges.data['d'] * edges.data['e_type'][:, 0].unsqueeze(1)
        , 'm_n2_v': edges.src['h'] * edges.data['w'] * edges.data['e_type'][:, 1].unsqueeze(1)
        , 'm_n2_w': edges.data['w'] * edges.data['e_type'][:, 1].unsqueeze(1)
        , 'm_n2_d': edges.data['d'] * edges.data['e_type'][:, 1].unsqueeze(1)
        }


def reduce(nodes):
    n1_v = torch.sum(nodes.mailbox['m_n1_v'], 1) / torch.sum(nodes.mailbox['m_n1_w'], 1)
    n2_v = torch.sum(nodes.mailbox['m_n2_v'], 1) / torch.sum(nodes.mailbox['m_n2_w'], 1)
    n1_e = nodes.mailbox['m_n1_d']
    n2_e = nodes.mailbox['m_n2_d']

    n1_h = torch.sum(nodes.mailbox['m_n1_h'], 1)
    n1_w = nodes.mailbox['m_n1_w']
    return {'n1_v': n1_v, 'n2_v': n2_v, 'n1_e': n1_e, 'n2_e': n2_e, 'n1_h': n1_h, 'n1_w': n1_w}


class NodeApplyModule(nn.Module):
    def __init__(self, k, m, ajr, hidden_dim, activation, use_x=True):
        super(NodeApplyModule, self).__init__()
        self.ajr = ajr
        self.m = m
        self.l0 = nn.Linear(2, hidden_dim)
        self.l1 = nn.Linear(k, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(ajr, hidden_dim)
        self.l5 = nn.Linear(m-1, hidden_dim)

        self.t3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.t4 = nn.Linear(1, hidden_dim, bias=False)

        self.activation = activation
        self.use_x = int(use_x)

    def forward(self, node):
        x = node.data['x']
        l = node.data['label']
        n1_v = node.data['n1_v']  # dist
        n2_v = node.data['n2_v']  # group
        n1_e = torch.sort(node.data['n1_e'], 1, descending=True)[0][:, 0: self.ajr].squeeze(2)
        n2_e = torch.sort(node.data['n2_e'], 1, descending=True)[0][:, 0: self.m-1].squeeze(2)

        n1_h = node.data['n1_h']
        n1_w = node.data['n1_w']  # weight
        h = self.activation(0
                            + self.l0(x) * self.use_x
                            + self.l1(l)
                            + self.l2(n1_h)
                            # + self.l3(n2_v)
                            + self.t3(torch.sum(self.activation(self.t4(n1_w)), dim=1))
                            # + self.l5(n2_e)
                            )

        # h = self.activation(self.l0(x) + self.l1(l)
        #                     + self.l2(n1_v)
        #                     # + self.l3(n2_v)
        #                     + self.l4(n1_w)
        #                     # + self.l5(n2_e)
        #                     )
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, k, m, ajr, hidden_dim, activation, use_x=True):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(k, m, ajr, hidden_dim, activation, use_x=use_x)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(udf_u_mul_e, reduce)
        g.apply_nodes(func=self.apply_mod)
        g.ndata.pop('n1_v')
        g.ndata.pop('n2_v')
        g.ndata.pop('n1_e')
        g.ndata.pop('n2_e')
        return g.ndata.pop('h')

# gnn = GCN(3, 3, 5, 7, F.relu)
# problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=7)
# g = problem.g
# h = gnn(g, g.ndata['h'])
# h = gnn(g, h)
# vis_g(problem, name='toy_models/gnn0', topo='knn')

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1, activate_linear=True):
        """
        :param h: num of head
        :param d_model: output dimension, h * hid_dim
        :param dropout:
        :param activate_linear:
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # d_v=d_k=hid_dim
        self.d_k = d_model // h  # 16
        self.h = h  # 4
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.activate_linear = activate_linear

    def clones(self, module, N):
        return nn.ModuleList([dc(module) for _ in range(N)])

    def attention(self, query, key, value, mask=None, dropout=None):
        # query: (b, h, 1, d)
        # key, value: (1, h, n, d) 1, h, d, n
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        # query: (bg, b, 4d)
        # key, value: (bg, n, 4d)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches_g = query.size(0)
        nbatches = query.size(1)
        if self.activate_linear:
            query, = [l(x).view(nbatches_g, nbatches, -1, self.h, self.d_k).transpose(2, 3) for l, x in zip(self.linears, (query, ))]
            key, value = [l(x).view(nbatches_g, 1, -1, self.h, self.d_k).transpose(2, 3) for l, x in zip(self.linears, (key, value))]
        else:
            query, = [x.view(nbatches_g, nbatches, -1, self.h, self.d_k).transpose(2, 3) for l, x in zip(self.linears, (query, ))]
            key, value = [x.view(nbatches_g, 1, -1, self.h, self.d_k).transpose(2, 3) for l, x in zip(self.linears, (key, value))]

        # query: (bg, b, h, 1, d)
        # key, value: (bg, 1, h, n, d)

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous().view(nbatches_g, nbatches, -1, self.h * self.d_k)
        if self.activate_linear:
            return self.linears[-1](x)
        else:
            return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.cuda()
        # self.register_buffer('pe', pe)

    def forward(self, x, position):
        x = x + self.pe[position]
        return self.dropout(x)

# pe = PositionalEncoding(10, 0, 50)
# x = torch.tensor([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
# x = pe(x, 1)
# problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
# m = DQNet(k=3, m=3, ajr=5, num_head=4, hidden_dim=16, extended_h=True, use_x=False).cuda()
# #
# g1 = to_cuda(problem.g)
# a1 = problem.get_legal_actions()
# problem.reset()
# g2 = to_cuda(problem.g)
# a2 = problem.get_legal_actions()
# problem.reset()
# g3 = to_cuda(problem.g)
# a3 = problem.get_legal_actions()
# g = dgl.batch([g1,g2,g3])
# actions = torch.cat([a1,a2,a3], axis=0)
#
# S_a_encoding1, h11, h21, Q_sa1 = m.forward_mha(g=dgl.batch([g2]), actions=a2.cuda())
#
# S_a_encoding, h1, h2, Q_sa = m.forward_mha(g=g, actions=actions.cuda())


#
# q = torch.rand((2, 76)).repeat(3,1,1).cuda()  #(b, 4h)
#
# key = torch.cat([g1.ndata['h'].repeat(1, 1, 4), g2.ndata['h'].repeat(1, 1, 4), g3.ndata['h'].repeat(1, 1, 4)], axis=0) #(n, 4h)
# value = key  #(n, 4h)
# t = m.MHA(q, key, value)  #(b, 1, 4h)

# q1 = q[0:2,:,:]
# key1 = key[0:2,:,:]
# value1 = value[0:2,:,:]
# t1 = m.MHA(q1, key1, value1)  #(b, 1, 4h)


class DQNet(nn.Module):
    # TODO k, m, ajr should be excluded::no generalization
    def __init__(self, k, m, ajr, num_head, hidden_dim, extended_h=False, use_x=True):
        super(DQNet, self).__init__()
        self.k = k
        self.m = m
        self.n = k * m
        self.num_head = num_head
        self.extended_h = extended_h
        self.hidden_dim = hidden_dim
        if self.extended_h:
            self.hidden_dim += k
        self.value1 = nn.Linear(num_head*self.hidden_dim, hidden_dim//2)
        self.value2 = nn.Linear(hidden_dim//2, 1)
        self.layers = nn.ModuleList([GCN(k, m, ajr, hidden_dim, F.relu, use_x=use_x)])
        self.MHA = MultiHeadedAttention(h=num_head
                           , d_model=num_head*self.hidden_dim
                           , dropout=0.0
                           , activate_linear=True)
        # baseline
        self.t5 = nn.Linear(2 * self.hidden_dim, 1)
        self.t6 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # # for centroid graph representation
        # self.t5_ = nn.Linear((self.k + 2) * self.hidden_dim, 1)
        # self.t6_ = nn.Linear(self.hidden_dim * self.k, self.hidden_dim * self.k)

        self.t7 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.t8 = nn.Linear(self.hidden_dim + self.k, self.hidden_dim)

        self.h_residual = []

    def forward_vanilla(self, g, actions=None, action_type='swap', gnn_step=3, time_aware=False, remain_episode_len=None):
        if isinstance(g, BatchedDGLGraph):
            batch_size = g.batch_size
            num_action = actions.shape[0] // batch_size
        else:
            num_action = actions.shape[0]


        h = g.ndata['h']

        # if in time-aware mode
        if time_aware:
            h[:, -1] += remain_episode_len

        for i in range(gnn_step):
            for conv in self.layers:
                h = conv(g, h)

        # if use extend h
        if self.extended_h:
            g.ndata['h'] = torch.cat([h, g.ndata['label']], dim=1)
        else:
            g.ndata['h'] = h

        h_new = h

        # compute group centroid for batch graph g
        batch_centroid = []
        for i in range(self.k):
            blocks_i = g.ndata['h'][(g.ndata['label'][:, i] > .5).nonzero().squeeze()]
            batch_centroid.append(torch.mean(blocks_i.view(batch_size, self.m, self.hidden_dim), axis=1))

        if isinstance(g, BatchedDGLGraph):
            # graph_embedding = self.t6_(torch.cat(batch_centroid, axis=1)).repeat(1, num_action).view(num_action * batch_size, -1) # batch_size * (h * k)
            graph_embedding = self.t6(dgl.mean_nodes(g, 'h')).repeat(1, num_action).view(num_action * batch_size, -1)
        else:
            graph_embedding = self.t6(torch.mean(g.ndata['h'], dim=0)).repeat(num_action, 1)

        if isinstance(g, BatchedDGLGraph):
            action_mask = torch.tensor(range(0, self.n * batch_size, self.n))\
                .unsqueeze(1).expand(batch_size, 2)\
                .repeat(1, num_action)\
                .view(num_action * batch_size, -1)

            actions_ = actions + action_mask.cuda()
        else:
            actions_ = actions

        if action_type == 'flip':
            q_actions = self.t8(torch.cat([g.ndata['h'][actions_[:, 0], :], torch.nn.functional.one_hot(actions_[:, 1], g.ndata['label'].shape[1]).float()], axis=1))
        if action_type == 'swap':
            # q_actions = torch.cat([g.ndata['h'][actions_[:, 0], :], g.ndata['h'][actions_[:, 1], :]], axis=1)
            q_actions = self.t7(torch.cat([g.ndata['h'][actions_[:, 0], :], g.ndata['h'][actions_[:, 1], :]], axis=1))

        S_a_encoding = torch.cat([graph_embedding, q_actions], axis=1)

        Q_sa = self.t5(F.relu(S_a_encoding)).squeeze()
        # Q_sa = self.t5_(F.relu(S_a_encoding)).squeeze()
        # Q_sa = (Q_sa.view(n, n) + Q_sa.view(n, n).t()).view(n**2)

        return S_a_encoding, h_new, g.ndata['h'], Q_sa

    def forward(self, g, actions=None, action_type='swap', gnn_step=3, time_aware=False, remain_episode_len=None):
        n = self.n
        k = self.k
        m = self.m
        hidden_dim = self.hidden_dim
        num_head = self.num_head

        if isinstance(g, BatchedDGLGraph):
            batch_size = g.batch_size
            num_action = actions.shape[0] // batch_size
        else:
            num_action = actions.shape[0]

        h = g.ndata['h']

        for i in range(gnn_step):
            for conv in self.layers:
                h = conv(g, h)

        # if use extend h
        if self.extended_h:
            g.ndata['h'] = torch.cat([h, g.ndata['label']], dim=1)
        else:
            g.ndata['h'] = h


        h_new = h
        # pe = PositionalEncoding(h.shape[1], dropout=0, max_len=max_step)
        # g.ndata['h'] = torch.cat([pe(h, remain_step), g.ndata['x'], g.ndata['label']], dim=1)
        # g.ndata['h'] = torch.cat([h, g.ndata['x'], g.ndata['label']], dim=1)
        # g.ndata['h'] = h

        # compute centroid embedding c_i
        # gc_x = torch.mm(g.ndata['x'].t(), g.ndata['label']) / m
        # gc_h = torch.mm(g.ndata['h'].t(), g.ndata['label']) / m
        gc_h = torch.bmm(g.ndata['x'].view(batch_size, n, -1).transpose(1, 2), \
                         g.ndata['label'].view(batch_size, n, -1)) / m
        # gc_h: (bg, d, k)

        key = g.ndata['h'].repeat(1, num_head).view(batch_size, n, -1)
        value = key  #(bg, n, 4d)

        # action = (0, 1) # swap - (i, j)
        # # query head: (x_i, x_j, c_i, c_j)
        # head1 = g.ndata['h'][action[0]]
        # head2 = g.ndata['h'][action[1]]
        # head3 = torch.mm(g.ndata['label'][action[0]].unsqueeze(0), gc_h.t()).squeeze()
        # head4 = torch.mm(g.ndata['label'][action[1]].unsqueeze(0), gc_h.t()).squeeze()
        # query = torch.cat([head1, head2, head3, head4], axis=0).unsqueeze(0)
        # query_mirror = torch.cat([head2, head1, head4, head3], axis=0).unsqueeze(0)

        # feed the whole action space (bg, num_action, 4d)

        if isinstance(g, BatchedDGLGraph):
            action_mask = torch.tensor(range(0, self.n * batch_size, self.n))\
                .unsqueeze(1).expand(batch_size, 2)\
                .repeat(1, num_action)\
                .view(num_action * batch_size, -1)

            actions_ = actions + action_mask.cuda()
        else:
            actions_ = actions
        q_h = torch.cat([g.ndata['h'][actions_[:, 0], :], \
                       g.ndata['h'][actions_[:, 1], :]], axis=1).view(batch_size, num_action, -1)

        q_c = q_h

        Q = torch.cat([q_h, q_c], axis=2)

        S_a_encoding = self.MHA(Q, key, value)# (bg, num_action, 1, 4d)
        # mN = dgl.mean_nodes(g, 'h')
        # PI = self.policy(g.ndata['h'])
        Q_sa = self.value2(F.relu(self.value1(S_a_encoding)))
        # g.ndata.pop('h')
        return S_a_encoding, h, g.ndata['h'], Q_sa.squeeze()
