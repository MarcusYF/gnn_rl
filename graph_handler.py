from copy import deepcopy as dc
import dgl
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from dataclasses import dataclass
# from collections import namedtuple
# Interval = namedtuple('extend_info', ['kcut_value', 'ub'])


@dataclass
class LightGraph:

    in_cuda: bool                   # if cuda graph
    n: int                          # number of nodes
    k: int                          # number of clusters
    m: list                         # number of nodes in each cluster
    batch_size: int                 # batch graph size
    ndata: dict                     # node attributes
    edata: dict                     # edge attributes
    kcut_value: torch.tensor        # current K-cut value
    # extend_info: tuple

    def number_of_nodes(self):
        return self.n

    def number_of_edges(self):
        return self.n * (self.n - 1)


def to_cuda(G_, copy=True):
    if copy:
        G = dc(G_)
    else:
        G = G_
    for node_attr in G.ndata.keys():
        G.ndata[node_attr] = G.ndata[node_attr].cuda()
    for edge_attr in G.edata.keys():
        G.edata[edge_attr] = G.edata[edge_attr].cuda()
    return G


class GraphGenerator:

    def __init__(self, k, m, ajr, style='plain'):
        self.k = k
        self.m = m
        if isinstance(m, list):
            self.n = sum(m)
            assert len(m) == k
            self.cut = 'unequal'
            self.init_label = []
            for i in range(k):
                self.init_label.extend([i]*m[i])
        else:
            self.n = k * m
            self.cut = 'equal'
        self.ajr = ajr
        self.nonzero_idx = [i for i in range(self.n ** 2) if i % (self.n + 1) != 0]
        self.adj_mask = torch.tensor(range(0, self.n ** 2, self.n)).unsqueeze(1).expand(self.n, ajr + 1)
        self.style = style

    def generate_graph(self, x=None, batch_size=1, style=None, seed=None, cuda_flag=True):

        k = self.k
        m = self.m
        n = self.n
        ajr = self.ajr
        if style is not None:
            style = style
        else:
            style = self.style

        if style.startswith('er'):
            p = float(style.split('-')[1])
            G = [nx.erdos_renyi_graph(n, p) for _ in range(batch_size)]
            adj_matrices = torch.cat([torch.tensor(nx.adjacency_matrix(g).todense()).float() for g in G])

        elif style.startswith('ba'):
            _m = int(style.split('-')[1])
            G = [nx.barabasi_albert_graph(n, _m) for _ in range(batch_size)]
            adj_matrices = torch.cat([torch.tensor(nx.adjacency_matrix(g).todense()).float() for g in G])

        # init batch graphs
        bg = LightGraph(cuda_flag, self.n, self.k, self.m, batch_size, {}, {}, torch.zeros(batch_size))

        # assign 2-d coordinates 'x'
        if x is None:
            if style == 'plain':
                if seed is not None:
                    np.random.seed(seed)
                    bg.ndata['x'] = torch.tensor(np.random.rand(batch_size * n, 2)).float()
                else:
                    bg.ndata['x'] = torch.rand((batch_size * n, 2))
            elif style == 'shift':
                bg.ndata['x'] = torch.rand((batch_size * n, 2)) * 10 + 5
            elif style.startswith('cluster'):
                _h = 2
                cluster_style = int(style.split('-')[1])
                if cluster_style == 0:
                    center = torch.rand((batch_size * k, 1, _h)).repeat(1, m, 1) * 6
                elif cluster_style == 1:  # k=4
                    mask = torch.tensor([[[0, 0]], [[0, 10]], [[10, 0]], [[10, 10]]]).repeat(batch_size, 1, 1)
                    center = torch.rand((batch_size * k, 1, _h)) * 3 + mask
                    center = center.repeat(1, m, 1)
                elif cluster_style == 2:  # k=4
                    mask = torch.tensor([[[0, 0]], [[0, 0]], [[10, 10]], [[10, 10]]]).repeat(batch_size, 1, 1)
                    center = torch.rand((batch_size * k, 1, _h)) * 3 + mask
                    center = center.repeat(1, m, 1)

                bg.ndata['x'] = (center + torch.rand((batch_size * k, m, _h))).view(batch_size * n, _h)
        else:
            bg.ndata['x'] = x

        # label
        if self.cut == 'equal':
            label = torch.tensor(range(k)).unsqueeze(1).repeat(batch_size, m).view(-1)
        else:
            label = torch.tensor(self.init_label).repeat(batch_size)
        batch_mask = torch.tensor(range(0, n * batch_size, n)).unsqueeze(1).expand(batch_size, n).flatten()
        if seed is not None:
            perm_idx = torch.cat([torch.tensor(np.random.permutation(n)) for _ in range(batch_size)]) + batch_mask
        else:
            perm_idx = torch.cat([torch.randperm(n) for _ in range(batch_size)]) + batch_mask
        label = label[perm_idx].view(batch_size, n)
        bg.ndata['label'] = torch.nn.functional.one_hot(label, k).float().view(batch_size * n, k)

        # d/adj
        if style.startswith('er') or style.startswith('ba'):
            bg.edata['d'] = adj_matrices.view(batch_size, -1, 1)[:, self.nonzero_idx, :].view(-1, 1)
        else:
            _, neighbor_idx, square_dist_matrix = dgl.transform.knn_graph(bg.ndata['x'].view(batch_size, n, -1), ajr + 1, extend_info=True)
            square_dist_matrix = F.relu(square_dist_matrix, inplace=True)  # numerical error could result in NaN in sqrt. value
            bg.ndata['adj'] = torch.sqrt(square_dist_matrix).view(bg.n * bg.batch_size, -1)
            # scale d (maintain avg=0.5):
            if style != 'plain':
                bg.ndata['adj'] /= (bg.ndata['adj'].sum() / (bg.ndata['adj'].shape[0]**2) / 0.5)
            bg.edata['d'] = bg.ndata['adj'].view(batch_size, -1, 1)[:, self.nonzero_idx, :].view(-1, 1)

        # e_type
        group_matrix = torch.bmm(bg.ndata['label'].view(batch_size, n, -1), bg.ndata['label'].view(batch_size, n, -1).transpose(1, 2)).view(batch_size, -1)[:, self.nonzero_idx].view(-1, 1)

        if style.startswith('er') or style.startswith('ba'):
            bg.edata['e_type'] = torch.cat([bg.edata['d'], group_matrix], dim=1)
        else:
            neighbor_idx -= torch.tensor(range(0, batch_size * n, n)).view(batch_size, 1, 1).repeat(1, n, ajr + 1) \
                            - torch.tensor(range(0, batch_size * n * n, n * n)).view(batch_size, 1, 1).repeat(1, n,
                                                                                                              ajr + 1)
            adjacent_matrix = torch.zeros((batch_size * n * n, 1))
            adjacent_matrix[neighbor_idx + self.adj_mask.repeat(batch_size, 1, 1)] = 1
            adjacent_matrix = adjacent_matrix.view(batch_size, n * n, 1)[:, self.nonzero_idx, :].view(-1, 1)
            bg.edata['e_type'] = torch.cat([adjacent_matrix, group_matrix], dim=1)

        if cuda_flag:
            to_cuda(bg, copy=False)
        # kcut value
        bg.kcut_value = calc_S(bg)
        return bg


def calc_S(states, mode='complete'):
    S = states.edata['e_type'][:, 1] * states.edata['d'][:, 0]
    if mode != 'complete':
        S *= states.edata['e_type'][:, 0]
    return S.view(states.batch_size, -1).sum(dim=1) / 2


def make_batch(graphs):

    bg = LightGraph(graphs[0].in_cuda
                    , graphs[0].number_of_nodes()
                    , graphs[0].k
                    , graphs[0].m
                    , len(graphs) * graphs[0].batch_size
                    , {}, {}, torch.zeros(len(graphs)))

    for node_attr in graphs[0].ndata.keys():
        bg.ndata[node_attr] = torch.cat([g.ndata[node_attr] for g in graphs])
    for edge_attr in graphs[0].edata.keys():
        bg.edata[edge_attr] = torch.cat([g.edata[edge_attr] for g in graphs])
    bg.kcut_value = torch.cat([g.kcut_value for g in graphs])

    return bg


def un_batch(graphs, copy=True):

    n = graphs.number_of_nodes()
    e = graphs.number_of_edges()
    batch_size = graphs.batch_size

    ndata = {}.fromkeys(graphs.ndata.keys())
    edata = {}.fromkeys(graphs.edata.keys())
    if copy:
        kcut_value = graphs.kcut_value.clone()
    else:
        kcut_value = graphs.kcut_value

    for node_attr in graphs.ndata.keys():
        if copy:
            ndata[node_attr] = [graphs.ndata[node_attr][i*n:(i+1)*n, :].clone() for i in range(batch_size)]
        else:
            ndata[node_attr] = [graphs.ndata[node_attr][i * n:(i + 1) * n, :] for i in range(batch_size)]
    for edge_attr in graphs.edata.keys():
        if copy:
            edata[edge_attr] = [graphs.edata[edge_attr][i*e:(i+1)*e, :].clone() for i in range(batch_size)]
        else:
            edata[edge_attr] = [graphs.edata[edge_attr][i * e:(i + 1) * e, :] for i in range(batch_size)]
    graph_list = [LightGraph(graphs.in_cuda
                             , n
                             , graphs.k
                             , graphs.m
                             , 1
                             , dict([(n_attr, ndata[n_attr][i]) for n_attr in graphs.ndata.keys()])
                             , dict([(e_attr, edata[e_attr][i]) for e_attr in graphs.edata.keys()])
                             , kcut_value[i:i+1]) for i in range(batch_size)]

    return graph_list


def perm_weight(graphs, eps=0.1):
    graphs.edata['d'] *= F.relu(torch.ones(graphs.edata['d'].shape).cuda() + eps * torch.randn(graphs.edata['d'].shape).cuda())


def reset_label(graphs, label, compute_S=True, rewire_edges=True):

    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    label = graphs.ndata['label'] = torch.nn.functional.one_hot(label, graphs.k).float()
    if graphs.in_cuda:
        graphs.ndata['label'] = label.cuda()
    else:
        graphs.ndata['label'] = label

    if rewire_edges:
        nonzero_idx = [i for i in range(graphs.n ** 2) if i % (graphs.n + 1) != 0]
        graphs.edata['e_type'][:, 1] = torch.bmm(graphs.ndata['label'].view(graphs.batch_size, graphs.n, -1)
                                             , graphs.ndata['label'].view(graphs.batch_size, graphs.n, -1).transpose(1, 2)) \
                                       .view(graphs.batch_size, -1)[:, nonzero_idx].view(-1)
    if compute_S:
        graphs.kcut_value = calc_S(graphs)
    return graphs

# G = GraphGenerator(3,[3,4,5],8, style='plain')
# g1 = G.generate_graph(batch_size=1, cuda_flag=True)
# g2 = G.generate_graph(batch_size=1, cuda_flag=True)
# gg = make_batch([g1, g2])
# x = un_batch(gg, copy=True)
# x[1].edata['e_type'] - g2.edata['e_type']
# x[0].ndata['x'] *=0
# gg.ndata['x']
# reset_label(g1, [0,1,2,0,1,2,0,1,2])
# reset_label(g2, [0,1,2,0,1,2,0,1,2])
# G=GraphGenerator(3,3,8, style='plain')
# g=G.generate_batch_G(batch_size=2)