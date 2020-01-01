import math
from copy import deepcopy as dc
import itertools
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_G(k, m, adjacent_reserve, hidden_dim, random_init_label=True, a=1, sample=True, label=None, x=None):

    g = dgl.DGLGraph()
    g.add_nodes(k*m)
    # 2-d coordinates 'x'
    if sample:
        # g.ndata['x'] = torch.tensor(
        #     [[0, 0], [0, 1], [1, 0], [1, 1], [0.7, 0.8], [1.5, 1.5], [1.5, 3], [3, 3], [3, 1.5], [1.8, 1.7], [1.8, 0.3],
        #      [1.8, 0.8], [2.3, 0.8], [2.3, 0.3], [2.1, 0.5]])
        g.ndata['x'] = x
    else:
        g.ndata['x'] = torch.rand((k*m, 2))

    if random_init_label:
        label = torch.tensor(range(k)).unsqueeze(1).expand(k, m).flatten()
        label = label[torch.randperm(k * m)]
    else:
        label = torch.tensor(label)
    g.ndata['label'] = torch.nn.functional.one_hot(label, k).float()
    # g.edata['e_type'] = torch.zeros((g.number_of_edges(), 1))
    # g.edata['d'] = torch.zeros((g.number_of_edges(), 1))
    # g.edata['w'] = torch.ones((g.number_of_edges(), 1))
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
    g.ndata['h'] = torch.rand((g.number_of_nodes(), hidden_dim))

    G = {'g':g, 'k':k, 'm':m, 'adjacent_reserve':adjacent_reserve, 'hidden_dim':hidden_dim, 'a':a}
    return G


class KCut_DGL():

    def __init__(self, k, m, adjacent_reserve, hidden_dim, random_init_label=True, a=1, label=None):
        self.g = generate_G(k, m, adjacent_reserve, hidden_dim, random_init_label=random_init_label, a=a, sample=False, label=label)['g']
        self.N = k * m
        self.k = k # num of clusters
        self.m = m # num of nodes in cluster
        self.adjacent_reserve = adjacent_reserve
        self.hidden_dim = hidden_dim
        self.random_init_label = random_init_label
        self.a = a
        self.S = self.calc_S()

    def get_graph_dims(self):
        return 2

    def calc_S(self, g=None):
        if g is None:
            g = self.g
        S = 0
        for i in range(self.k):
            block_i = g.ndata['x'][(g.ndata['label'][:, i] > 0).nonzero().squeeze()]
            for j in range(self.m):
                block_ij = block_i - block_i[j]
                S += torch.sum(torch.sqrt(torch.diag(torch.mm(block_ij, block_ij.t()))))
        return S/2

    def reset(self):
        self.g = generate_G(self.k, self.m, self.adjacent_reserve, self.hidden_dim, random_init_label=self.random_init_label, a=self.a, sample=False)['g']
        self.S = self.calc_S()
        return self.g

    def reset_label(self, label):
        self.g = generate_G(self.k, self.m, self.adjacent_reserve, self.hidden_dim, random_init_label=False, a=self.a, sample=True, label=label, x=self.g.ndata['x'])['g']
        self.S = self.calc_S()
        return self.g

    def get_legal_actions(self):
        return list(itertools.product(range(self.N), range(self.N)))

    def calc_swap_delta(self, i, j, i_label, j_label, state=None):

        if state is None:
            state = self.g

        return state.ndata['x'][i] - state.ndata['x'][(state.ndata['label'][:, i_label] > 0).nonzero().squeeze()], \
                state.ndata['x'][j] - state.ndata['x'][(state.ndata['label'][:, j_label] > 0).nonzero().squeeze()]

    def step(self, action, state=None):

        mode = 'outer'
        if state is None:
            mode = 'self'
            state = self.g

        i, j = action
        i_label, j_label = [(state.ndata['label'][c] > 0).nonzero().item() for c in action]

        if i == j or i_label == j_label:
            return state, torch.tensor(0.0)

        # rewire edges
        group_i = (state.ndata['label'][:, i_label] > 0).nonzero().squeeze()
        group_j = (state.ndata['label'][:, j_label] > 0).nonzero().squeeze()
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
        reward = torch.sqrt(torch.sum(torch.pow(torch.cat([old_0, old_1]), 2), axis=1)).sum() - torch.sqrt(torch.sum(torch.pow(torch.cat([new_0, new_1]), 2), axis=1)).sum()


        if mode == 'self':
            self.S -= reward
            return self.g, reward
        else:
            return state, reward


def udf_u_mul_e(edges):
    # a= edges.data['d'] * edges.data['e_type'][:, 0].unsqueeze(1)
    # print(a.view(15,14,1))
    return {'m_n1_v': edges.src['h'] * edges.data['w'] * edges.data['e_type'][:, 0].unsqueeze(1)
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
    return {'n1_v': n1_v, 'n2_v': n2_v, 'n1_e': n1_e, 'n2_e': n2_e}


class NodeApplyModule(nn.Module):
    def __init__(self, k, m, ajr, hidden_dim, activation):
        super(NodeApplyModule, self).__init__()
        self.ajr = ajr
        self.m = m
        self.l0 = nn.Linear(2, hidden_dim)
        self.l1 = nn.Linear(k, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(ajr, hidden_dim)
        self.l5 = nn.Linear(m-1, hidden_dim)
        self.activation = activation

    def forward(self, node):
        x = node.data['x']
        l = node.data['label']
        n1_v = node.data['n1_v']
        n2_v = node.data['n2_v']
        # print(torch.sort(node.data['n2_e'], 1, descending=True)[0][:, 0: self.m-1].squeeze(2))
        n1_e = torch.sort(node.data['n1_e'], 1, descending=True)[0][:, 0: self.ajr].squeeze(2)
        n2_e = torch.sort(node.data['n2_e'], 1, descending=True)[0][:, 0: self.m-1].squeeze(2)
        h = self.activation(self.l0(x) + self.l1(l)
                            + self.l2(n1_v)
                            + self.l3(n2_v)
                            + self.l4(n1_e)
                            + self.l5(n2_e)
                            )
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, k, m, ajr, hidden_dim, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(k, m, ajr, hidden_dim, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(udf_u_mul_e, reduce)
        g.apply_nodes(func=self.apply_mod)
        g.ndata.pop('n1_v')
        g.ndata.pop('n2_v')
        g.ndata.pop('n1_e')
        g.ndata.pop('n2_e')
        return g.ndata.pop('h')


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
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.activate_linear = activate_linear

    def clones(self, module, N):
        return nn.ModuleList([dc(module) for _ in range(N)])

    def attention(self, query, key, value, mask=None, dropout=None):
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
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if self.activate_linear:
            query, = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, ))]
            key, value = [l(x).view(1, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (key, value))]
        else:
            query, = [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, ))]
            key, value = [x.view(1, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (key, value))]
        x, self.attn = self.attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
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


class DQNet(nn.Module):
    # TODO k, m, ajr should be excluded::no generalization
    def __init__(self, k, m, ajr, num_head, hidden_dim):
        super(DQNet, self).__init__()
        self.k = k
        self.m = m
        self.n = k * m
        self.num_head = num_head
        self.hidden_dim = hidden_dim + 2 + k
        self.value1 = nn.Linear(num_head*self.hidden_dim, hidden_dim//2)
        self.value2 = nn.Linear(hidden_dim//2, 1)
        self.layers = nn.ModuleList([GCN(k, m, ajr, hidden_dim, F.relu)])
        self.MHA = MultiHeadedAttention(h=num_head \
                           , d_model=num_head*self.hidden_dim \
                           , dropout=0.0 \
                           , activate_linear=True)

        self.h_residual = []

    def forward(self, g, gnn_step, max_step, remain_step):
        n = self.n
        k = self.k
        m = self.m
        hidden_dim = self.hidden_dim
        num_head = self.num_head

        h = g.ndata['h']
        self.h_residual = []
        for i in range(gnn_step):
            for conv in self.layers:
                h_new = conv(g, h)
                self.h_residual.append(torch.norm(h_new-h).detach())
                h = h_new

        pe = PositionalEncoding(h.shape[1], dropout=0, max_len=max_step)
        g.ndata['h'] = torch.cat([pe(h, remain_step), g.ndata['x'], g.ndata['label']], dim=1)

        # compute centroid embedding c_i
        # gc_x = torch.mm(g.ndata['x'].t(), g.ndata['label']) / m
        gc_h = torch.mm(g.ndata['h'].t(), g.ndata['label']) / m


        key = g.ndata['h'].repeat(1, num_head)
        value = g.ndata['h'].repeat(1, num_head)
        # action = (0, 1) # swap - (i, j)
        # # query head: (x_i, x_j, c_i, c_j)
        # head1 = g.ndata['h'][action[0]]
        # head2 = g.ndata['h'][action[1]]
        # head3 = torch.mm(g.ndata['label'][action[0]].unsqueeze(0), gc_h.t()).squeeze()
        # head4 = torch.mm(g.ndata['label'][action[1]].unsqueeze(0), gc_h.t()).squeeze()
        # query = torch.cat([head1, head2, head3, head4], axis=0).unsqueeze(0)
        # query_mirror = torch.cat([head2, head1, head4, head3], axis=0).unsqueeze(0)

        # feed the whole action space
        Q1 = torch.cat([g.ndata['h'].repeat(1, n).view(n * n, hidden_dim) \
                           , g.ndata['h'].repeat(n, 1) \
                           , torch.mm(g.ndata['label'].repeat(1, n).view(n * n, k), gc_h.t()) \
                           , torch.mm(g.ndata['label'].repeat(n, 1), gc_h.t())]
                       , axis=1)

        Q2 = torch.cat([Q1[:, hidden_dim:hidden_dim * 2]
                           , Q1[:, 0:hidden_dim]
                           , Q1[:, hidden_dim * 3:hidden_dim * 4]
                           , Q1[:, hidden_dim * 2:hidden_dim * 3]]
                       , axis=1)

        S_a_encoding = self.MHA(Q1, key, value) + self.MHA(Q2, key, value)
        # mN = dgl.mean_nodes(g, 'h')
        # PI = self.policy(g.ndata['h'])
        Q_sa = self.value2(F.relu(self.value1(S_a_encoding)))
        # g.ndata.pop('h')
        return S_a_encoding, h, g.ndata['h'], Q_sa.squeeze()
