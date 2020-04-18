import math
from copy import deepcopy as dc
import itertools
import dgl
from dgl.batched_graph import BatchedDGLGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import time


def is_identical(G1, G2):
    a = torch.max(torch.pow(G1.ndata['x'] - G2.ndata['x'], 2)).item() + \
    torch.max(torch.pow(G1.ndata['label'] - G2.ndata['label'], 2)).item() + \
    torch.max(torch.pow(G1.edata['d'] - G2.edata['d'], 2)).item() + \
    torch.max(torch.pow(G1.edata['e_type'] - G2.edata['e_type'], 2)).item()
    if a < 1e-8:
        return True
    else:
        print(a)
        return False


class GraphGenerator:

    def __init__(self, k, m, ajr, style='plain'):

        self.k = k
        self.m = m
        self.n = k * m
        self.ajr = ajr
        self.nonzero_idx = [i for i in range(self.n**2) if i % (self.n+1) != 0]
        self.src = [i // self.n for i in self.nonzero_idx]
        self.dst = [i % self.n for i in self.nonzero_idx]
        self.adj_mask = torch.tensor(range(0, self.n ** 2, self.n)).unsqueeze(1).expand(self.n, ajr + 1)
        self.style = style

    def generate_G(self, x=None, label=None, hidden_dim=16, a=1):

        # init graph
        k = self.k
        m = self.m
        n = self.n
        g = dgl.DGLGraph()

        g.add_nodes(n)

        # 2-d coordinates 'x'
        if x is None:
            g.ndata['x'] = torch.rand((n, 2))
        else:
            g.ndata['x'] = x

        if label is None:
            label = torch.tensor(range(k)).unsqueeze(1).expand(k, m).flatten()
            label = label[torch.randperm(n)]
        else:
            label = torch.tensor(label)

        g.ndata['label'] = torch.nn.functional.one_hot(label, k).float()

        _, neighbor_idx, dist_matrix = dgl.transform.knn_graph(g.ndata['x'], self.ajr + 1, extend_info=True)

        g.add_edges(self.src, self.dst)
        g.edata['d'] = torch.sqrt(dist_matrix[0].view(-1, 1)[self.nonzero_idx, :])
        g.edata['w'] = 1.0 / (1.0 + torch.exp(a * g.edata['d']))

        adjacent_matrix = torch.zeros((n * n, 1))
        adjacent_matrix[neighbor_idx[0] + self.adj_mask] = 1

        group_matrix = torch.mm(g.ndata['label'], g.ndata['label'].t()).view(-1, 1)

        g.edata['e_type'] = torch.cat([adjacent_matrix[self.nonzero_idx, :], group_matrix[self.nonzero_idx, :]], dim=1)

        # init node embedding h
        g.ndata['h'] = torch.zeros((g.number_of_nodes(), hidden_dim))

        return g

    def generate_batch_G(self, target_bg=None, x=None, batch_size=1, hidden_dim=16, a=1, style=None):

        # init graph
        k = self.k
        m = self.m
        n = self.n
        ajr = self.ajr
        if style is not None:
            style = style
        else:
            style = self.style
        if target_bg is not None:
            bg = dgl.batch(np.random.choice(target_bg, batch_size, replace=True))
        else:
            if style.startswith('er'):
                p = float(style.split('-')[1])
                G = [nx.erdos_renyi_graph(n, p) for _ in range(batch_size)]
                adj_matrices = torch.cat([torch.tensor(nx.adjacency_matrix(g).todense()).float() for g in G])

            elif style.startswith('ba'):
                _m = int(style.split('-')[1])
                G = [nx.barabasi_albert_graph(n, _m) for _ in range(batch_size)]
                adj_matrices = torch.cat([torch.tensor(nx.adjacency_matrix(g).todense()).float() for g in G])

            # init batch graphs
            gs = [dgl.DGLGraph() for _ in range(batch_size)]
            _ = [(g.add_nodes(n), g.add_edges(self.src, self.dst)) for g in gs]

            bg = dgl.batch(gs)

            # 2-d coordinates 'x'
            if x is None:
                if style == 'plain':
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
        label = torch.tensor(range(k)).unsqueeze(1).repeat(batch_size, m).view(-1)
        batch_mask = torch.tensor(range(0, n * batch_size, n)).unsqueeze(1).expand(batch_size, n).flatten()
        perm_idx = torch.cat([torch.randperm(n) for _ in range(batch_size)]) + batch_mask
        label = label[perm_idx].view(batch_size, n)
        bg.ndata['label'] = torch.nn.functional.one_hot(label, k).float().view(batch_size * n, k)

        # calculate edges
        if target_bg is not None:
            # permute the dist matrix
            bg.edata['d'] *= F.relu(torch.ones(bg.edata['d'].shape).cuda() + 0.1 * torch.randn(bg.edata['d'].shape).cuda())
        else:
            if style.startswith('er') or style.startswith('ba'):
                bg.edata['d'] = adj_matrices.view(batch_size, -1, 1)[:, self.nonzero_idx, :].view(-1, 1)
            else:
                _, neighbor_idx, square_dist_matrix = dgl.transform.knn_graph(bg.ndata['x'].view(batch_size, n, -1), ajr + 1, extend_info=True)
                square_dist_matrix = F.relu(square_dist_matrix, inplace=True)  # numerical error could result in NaN in sqrt. value
                bg.edata['d'] = torch.sqrt(square_dist_matrix.view(batch_size, -1, 1)[:, self.nonzero_idx, :]).view(-1, 1)
                # scale d (maintain avg=0.5):
                if style != 'plain':
                    bg.edata['d'] /= (bg.edata['d'].sum() / bg.edata['d'].shape[0] / 0.5)

        group_matrix = torch.bmm(bg.ndata['label'].view(batch_size, n, -1), bg.ndata['label'].view(batch_size, n, -1).transpose(1, 2)).view(batch_size, -1)[:, self.nonzero_idx].view(-1, 1)

        if target_bg is not None:
            bg.edata['e_type'][:, 1:] = group_matrix
        else:
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

        return bg


class KCut_DGL:

    def __init__(self, k, m, adjacent_reserve, hidden_dim, mode='complete', x=None, label=None, a=1, graph_style='plain', sample_episode=10):
        self.graph_generator = GraphGenerator(k=k, m=m, ajr=adjacent_reserve, style=graph_style)
        self.g = self.graph_generator.generate_G(x=x, label=label, hidden_dim=hidden_dim, a=a)
        self.N = k * m
        self.k = k  # num of clusters
        self.m = m  # num of nodes in cluster
        self.adjacent_reserve = adjacent_reserve
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.x = x
        self.label = label
        self.a = a
        self.S = self.calc_S()
        self.sample_episode = sample_episode
        self.gen_step_batch_mask()

    def gen_step_batch_mask(self):
        # assert(sample_episode == states.batch_size) in step_batch
        self.mask1 = torch.tensor(range(0, self.N * self.sample_episode, self.N)).cuda()
        self.mask2 = torch.tensor(range(0, self.N * self.sample_episode, self.N)).repeat(self.m, 1).t().flatten().cuda()

    def calc_S(self, g=None):
        if g is None:
            g = self.g

        if self.mode == 'complete':
            S = torch.sum(g.edata['e_type'][:, 1] * g.edata['d'][:, 0])
        else:
            S = torch.sum((g.edata['e_type'].sum(dim=1) > 1.5) * g.edata['d'][:, 0])
        return S / 2

    def calc_batchS(self, bg, actions=None):
        if self.mode == 'complete':
            S = torch.sum((bg.edata['e_type'][:, 1] * bg.edata['d'][:, 0]).view(bg.batch_size, -1), dim=1)
        else:
            S = torch.sum(((bg.edata['e_type'].sum(dim=1) > 1.5) * bg.edata['d'][:, 0]).view(bg.batch_size, -1), dim=1)

        if actions is None:
            return S / 2
        else:
            actions.view(bg.batch_size, -1, 2)  # (b, num_action, 2)

    def gen_batch_graph(self, x=None, batch_size=1, hidden_dim=16, a=1, style=None):
        return self.graph_generator.generate_batch_G(x=x, batch_size=batch_size, hidden_dim=hidden_dim, a=a, style=style)

    def gen_target_batch_graph(self, target_bg=None, x=None, batch_size=1, hidden_dim=16, a=1, style=None):
        return self.graph_generator.generate_batch_G(target_bg=target_bg, x=x, batch_size=batch_size, hidden_dim=hidden_dim, a=a, style=style)

    def reset(self, compute_S=True):
        self.g = self.graph_generator.generate_G(x=self.x, label=self.label, a=self.a)
        if compute_S:
            self.S = self.calc_S()
        return self.g

    def reset_label(self, label, g=None, calc_S=True, rewire_edges=True):
        if g is None:
            g = self.g
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        g.ndata['label'] = torch.nn.functional.one_hot(label, self.k).float().cuda()

        # rewire edges
        if rewire_edges:
            for i, j in itertools.product(range(self.N), range(self.N)):
                if j != i:
                    u_label = g.nodes[j].data['label']
                    v_label = g.nodes[i].data['label']
                    if torch.max(u_label-v_label) < .5:
                        cluster_label = torch.tensor([[1.]]).cuda()
                    else:
                        cluster_label = torch.tensor([[0.]]).cuda()
                    # add group members
                    g.edges[i, j].data['e_type'] \
                        = torch.cat([g.edges[i, j].data['e_type'][:, 0].unsqueeze(0), cluster_label], dim=1)

        if g is None and calc_S:
            self.S = self.calc_S()
        return g

    def set_batch_label(self, bg, label):
        bg.ndata['label'] = torch.nn.functional.one_hot(label, self.k).float().cuda()
        # rewire edges
        n = bg.ndata['label'].shape[0] // bg.batch_size
        nonzero_idx = [i for i in range(n ** 2) if i % (n + 1) != 0]
        bg.edata['e_type'][:, 1] = torch.bmm(bg.ndata['label'].view(bg.batch_size, n, -1),
                                                 bg.ndata['label'].view(bg.batch_size, n, -1).transpose(1,
                                                                                                         2)).view(
            bg.batch_size, -1)[:, nonzero_idx].view(-1)

    def get_legal_actions(self, state=None, update=False, action_type='swap', action_dropout=1.0, pause_action=True):

        if state is None:
            state = self.g

        if action_type == 'flip':
            legal_actions = torch.nonzero(1 - state.ndata['label'])
        if action_type == 'swap':
            if isinstance(state, BatchedDGLGraph):
                n = state.number_of_nodes() // state.batch_size
                mask = torch.bmm(state.ndata['label'].view(state.batch_size, n, -1),
                                 state.ndata['label'].view(state.batch_size, n, -1).transpose(1, 2))
                legal_actions = torch.triu(1 - mask).nonzero()[:, 1:3]  # tensor (270, 2)

                # action_dropout:
                if action_dropout < 1.0:
                    num_actions = legal_actions.shape[0] // state.batch_size
                    maintain_actions = int(num_actions * action_dropout)
                    maintain = [np.random.choice(range(_ * num_actions, (_ + 1) * num_actions), maintain_actions, replace=False) for _ in range(state.batch_size)]
                    legal_actions = legal_actions[torch.tensor(maintain).flatten(), :]
                if pause_action:

                    legal_actions = legal_actions.reshape(state.batch_size, -1, 2)
                    legal_actions = torch.cat([legal_actions, (legal_actions[:, 0] * 0).unsqueeze(1)], dim=1).view(-1, 2)

            else:
                mask = torch.mm(state.ndata['label'], state.ndata['label'].t())
                legal_actions = torch.triu(1 - mask).nonzero()  # List[tensor(27, 2), ...]
                # action_dropout:
                if action_dropout < 1.0:
                    num_actions = legal_actions.shape[0]
                    maintain_actions = int(num_actions * action_dropout)
                    maintain = np.random.choice(range(num_actions), maintain_actions, replace=False)
                    legal_actions = legal_actions[torch.tensor(maintain).flatten(), :]
                if pause_action:
                    # add a special pause action (0, 0)
                    legal_actions = torch.cat([legal_actions, (legal_actions[0] * 0).unsqueeze(0)], dim=0)

            return legal_actions

    def calc_swap_delta(self, i, j, i_label, j_label, state=None, n=9, batch_size=10):

        if state is None:
            state = self.g

        if isinstance(state, BatchedDGLGraph):
            # i = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
            # j = torch.tensor([1, 3, 4, 5, 7, 8, 2, 4, 6, 7])
            # i_label = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
            # j_label = [1, 1, 0, 1, 0, 0, 2, 0, 2, 0]
            xi, xj = state.ndata['x'][i + self.mask1].view(batch_size, 1, 2), state.ndata['x'][j + self.mask1].view(batch_size, 1, 2)
            group_position_ = torch.nn.functional.one_hot(torch.tensor([i_label, j_label]).t(), self.k).float().transpose(1, 2).cuda()
            group_position = torch.bmm(state.ndata['label'].view(batch_size, n, self.k), group_position_)
            xi_group, xj_group = state.ndata['x'][group_position[:, :, 0].nonzero()[:, 1] + self.mask2].view(batch_size, self.m, 2)\
                , state.ndata['x'][group_position[:, :, 1].nonzero()[:, 1] + self.mask2].view(batch_size, self.m, 2)

            return xi - xi_group, xj - xj_group  #(batch_size, m, 2)
        else:
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

        if self.mode == 'complete':
            if action_type == 'swap':
                i, j = action
                i_label, j_label = [(state.ndata['label'][c] > 0).nonzero().item() for c in action]

                if i == j or i_label == j_label:
                    return state, torch.tensor(.0)


                # if rewire_edges:
                #     group_i = (state.ndata['label'][:, i_label] > .5).nonzero().squeeze()
                #     group_j = (state.ndata['label'][:, j_label] > .5).nonzero().squeeze()
                #     state.edges[i, group_i].data['e_type'] -= torch.tensor([[0, 1]]).repeat(self.m - 1, 1)
                #     state.edges[group_i, i].data['e_type'] -= torch.tensor([[0, 1]]).repeat(self.m - 1, 1)
                #     state.edges[i, group_j].data['e_type'] += torch.tensor([[0, 1]]).repeat(self.m, 1)
                #     state.edges[group_j, i].data['e_type'] += torch.tensor([[0, 1]]).repeat(self.m, 1)
                #     state.edges[j, group_j].data['e_type'] -= torch.tensor([[0, 1]]).repeat(self.m - 1, 1)
                #     state.edges[group_j, j].data['e_type'] -= torch.tensor([[0, 1]]).repeat(self.m - 1, 1)
                #     state.edges[j, group_i].data['e_type'] += torch.tensor([[0, 1]]).repeat(self.m, 1)
                #     state.edges[group_i, j].data['e_type'] += torch.tensor([[0, 1]]).repeat(self.m, 1)
                #     state.edges[j, i].data['e_type'] -= torch.tensor([[0, 2]])
                #     state.edges[i, j].data['e_type'] -= torch.tensor([[0, 2]])

                # compute reward
                old_0, old_1 = self.calc_swap_delta(i, j, i_label, j_label, state)

                # swap two nodes
                tmp = dc(state.nodes[i].data['label'])
                state.nodes[i].data['label'] = state.nodes[j].data['label']
                state.nodes[j].data['label'] = tmp

                # rewire edges
                if rewire_edges:
                    state.edata['e_type'][:, 1] = torch.mm(state.ndata['label'], state.ndata['label'].t()).view(-1)[
                        self.graph_generator.nonzero_idx]

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

        else:
            i, j = action
            i_label, j_label = [(state.ndata['label'][c] > 0).nonzero().item() for c in action]

            if i == j or i_label == j_label:
                return state, torch.tensor(.0)
            # compute old S
            remain_edges = (state.edata['e_type'].sum(dim=1) > 1.5) * state.edata['d'][:, 0]
            old_S = remain_edges.sum()
            # swap two nodes
            tmp = dc(state.nodes[i].data['label'])
            state.nodes[i].data['label'] = state.nodes[j].data['label']
            state.nodes[j].data['label'] = tmp
            # rewire edges
            state.edata['e_type'][:, 1] = torch.mm(state.ndata['label'], state.ndata['label'].t()).view(-1)[
                self.graph_generator.nonzero_idx]
            # compute new S
            remain_edges = (state.edata['e_type'].sum(dim=1) > 1.5) * state.edata['d'][:, 0]
            new_S = remain_edges.sum()

            reward = (old_S - new_S) / 2

        if state is None:
            self.S -= reward

        return state, reward

    def step_batch(self, states, action, return_sub_reward=False):
        """
        :param states: BatchedDGLGraph
        :param action: torch.tensor((batch_size, 2))
        :return:
        """
        batch_size = states.batch_size
        n = states.number_of_nodes() // batch_size

        if batch_size != self.sample_episode:
            self.sample_episode = batch_size
            self.gen_step_batch_mask()

        ii, jj = action[:, 0], action[:, 1]
        # label indexing is very time consuming
        # ij_label = [(states.ndata['label'][c] > 0).nonzero().item() for c in action.t().flatten() + torch.tensor(range(0, n * batch_size, n)).repeat(1, 2).squeeze(0).cuda()]
        # ii_label, jj_label = ij_label[0: batch_size], ij_label[batch_size: 2 * batch_size]

        if self.mode == 'complete':

            remain_edges = states.edata['e_type'][:, 1] * states.edata['d'][:, 0]
            old_S = remain_edges.view(batch_size, -1).sum(dim=1)
            # swap two sets of nodes

            tmp = dc(states.ndata['label'][ii + self.mask1])
            states.ndata['label'][ii + self.mask1] = states.ndata['label'][jj + self.mask1]
            states.ndata['label'][jj + self.mask1] = tmp
            # rewire edges
            states.edata['e_type'][:, 1] = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
                                                     states.ndata['label'].view(batch_size, n, -1).transpose(1,
                                                                                                             2)).view(
                batch_size, -1)[:, self.graph_generator.nonzero_idx].view(-1)

            # compute new S
            remain_edges = states.edata['e_type'][:, 1] * states.edata['d'][:, 0]
            new_S = remain_edges.view(batch_size, -1).sum(dim=1)

            rewards = (old_S - new_S) / 2

            # if True:
            #     # compute old S
            #     #  (b, k, n*(n-1))
            #     group_matrix_k = torch.bmm(
            #         states.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * self.k, n, 1),
            #         states.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * self.k, n, 1).transpose(1, 2)).view(batch_size * self.k, -1)[:, self.graph_generator.nonzero_idx].view(batch_size, self.k, -1)
            #     #  (k, b*n*(n-1))
            #     remain_edges = group_matrix_k.transpose(0, 1).reshape(self.k, -1) * states.edata['d'][:, 0]
            #     #  (k, b)
            #     old_S_k = remain_edges.view(self.k, batch_size, -1).sum(dim=2)
            #     #  (b)
            #     old_S = old_S_k.sum(dim=0)
            #
            #     # swap two sets of nodes
            #     tmp = dc(states.ndata['label'][ii + self.mask1])
            #     states.ndata['label'][ii + self.mask1] = states.ndata['label'][jj + self.mask1]
            #     states.ndata['label'][jj + self.mask1] = tmp
            #
            #     # rewire edges
            #     states.edata['e_type'][:, 1] = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
            #                                              states.ndata['label'].view(batch_size, n, -1).transpose(1,
            #                                                                                                      2)).view(
            #         batch_size, -1)[:, self.graph_generator.nonzero_idx].view(-1)
            #
            #     # compute new S
            #     #  (b, k, n*(n-1))
            #     group_matrix_k = torch.bmm(
            #         states.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * self.k, n, 1),
            #         states.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * self.k, n, 1).transpose(1, 2)).view(batch_size * self.k, -1)[:, self.graph_generator.nonzero_idx].view(batch_size, self.k, -1)
            #     #  (k, b*n*(n-1))
            #     remain_edges = group_matrix_k.transpose(0, 1).reshape(self.k, -1) * states.edata['d'][:, 0]
            #     #  (k, b)
            #     new_S_k = remain_edges.view(self.k, batch_size, -1).sum(dim=2)
            #     #  (b)
            #     new_S = new_S_k.sum(dim=0)
            #
            #     rewards = (old_S - new_S) / 2
            #
            #     sub_rewards = (old_S_k - new_S_k) / 2
            #
            # else:
            #     # compute reward  (batch_size, m, 2)
            #     old_0, old_1 = self.calc_swap_delta(ii, jj, ii_label, jj_label, states, n=n, batch_size=batch_size)
            #
            #     # swap two sets of nodes
            #     tmp = dc(states.ndata['label'][ii + self.mask1])
            #     states.ndata['label'][ii + self.mask1] = states.ndata['label'][jj + self.mask1]
            #     states.ndata['label'][jj + self.mask1] = tmp
            #     # pert_idx = torch.tensor(range(batch_size * n))
            #     # pert_idx[ii + self.mask1] = jj + self.mask1
            #     # pert_idx[jj + self.mask1] = ii + self.mask1
            #     # states.ndata['label'] = states.ndata['label'][pert_idx]
            #
            #     # rewire edges
            #     states.edata['e_type'][:, 1] = torch.bmm(states.ndata['label'].view(batch_size, n, -1), states.ndata['label'].view(batch_size, n, -1).transpose(1, 2)).view(batch_size, -1)[:, self.graph_generator.nonzero_idx].view(-1)
            #
            #     new_0, new_1 = self.calc_swap_delta(ii, jj, jj_label, ii_label, states, n=n, batch_size=batch_size)
            #
            #     rewards = torch.sqrt(torch.sum(torch.pow(torch.cat([old_0, old_1], axis=1), 2), axis=2)).sum(dim=1) \
            #              - torch.sqrt(torch.sum(torch.pow(torch.cat([new_0, new_1], axis=1), 2), axis=2)).sum(dim=1)
        else:
            # compute old S
            remain_edges = (states.edata['e_type'].sum(dim=1) > 1.5) * states.edata['d'][:, 0]
            old_S = remain_edges.view(batch_size, -1).sum(dim=1)
            # swap two sets of nodes
            tmp = dc(states.ndata['label'][ii + self.mask1])
            states.ndata['label'][ii + self.mask1] = states.ndata['label'][jj + self.mask1]
            states.ndata['label'][jj + self.mask1] = tmp

            # rewire edges
            states.edata['e_type'][:, 1] = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
                                                     states.ndata['label'].view(batch_size, n, -1).transpose(1, 2)).view(batch_size, -1)[:, self.graph_generator.nonzero_idx].view(-1)
            # compute new S
            remain_edges = (states.edata['e_type'].sum(dim=1) > 1.5) * states.edata['d'][:, 0]
            new_S = remain_edges.view(batch_size, -1).sum(dim=1)

            rewards = (old_S - new_S) / 2


        return states, rewards

def udf_u_mul_e(edges):
    # a= edges.data['d'] * edges.data['e_type'][:, 0].unsqueeze(1)
    # print(a.view(15,14,1))
    # print('h shape:', edges.src['h'].shape)
    # print('d shape:', edges.data['d'].shape)
    return {
        'm_n1_h': edges.src['h'] * edges.data['e_type'][:, 0].unsqueeze(1)
        , 'm_n2_h': edges.src['h'] * edges.data['e_type'][:, 1].unsqueeze(1)
        , 'm_n_l': edges.src['label']
        # , 'm_n1_hd': torch.cat([edges.src['h'], edges.data['d']], dim=1) * edges.data['e_type'][:, 0].unsqueeze(1)
        # , 'm_n1_v': edges.src['h'] * edges.data['w'] * edges.data['e_type'][:, 0].unsqueeze(1)
        # , 'm_n1_w': edges.data['w'] * edges.data['e_type'][:, 0].unsqueeze(1)
        , 'm_n1_d': edges.data['d'] * edges.data['e_type'][:, 0].unsqueeze(1)
        # , 'm_n_w': edges.data['w']
        # , 'm_n_d': edges.data['d']
        # , 'm_n2_v': edges.src['h'] * edges.data['w'] * edges.data['e_type'][:, 1].unsqueeze(1)
        # , 'm_n2_w': edges.data['w'] * edges.data['e_type'][:, 1].unsqueeze(1)
        , 'm_n2_d': edges.data['d'] * edges.data['e_type'][:, 1].unsqueeze(1)
    }

def reduce(nodes):
    # n1_v = torch.sum(nodes.mailbox['m_n1_v'], 1) / torch.sum(nodes.mailbox['m_n1_w'], 1)
    # n2_v = torch.sum(nodes.mailbox['m_n2_v'], 1) / torch.sum(nodes.mailbox['m_n2_w'], 1)
    # n1_e = nodes.mailbox['m_n1_d']
    # n2_e = nodes.mailbox['m_n2_d']

    # n1_h = torch.sum(nodes.mailbox['m_n1_h'], 1)
    # n1_w = nodes.mailbox['m_n1_w']
    n1_hd = nodes.mailbox['m_n1_h']
    n2_hd = nodes.mailbox['m_n2_h']
    # nodes.mailbox['m_n1_d']: of size (n*b, n-1, 1)
    n1_d = nodes.mailbox['m_n1_d']
    # n1_d = (nodes.mailbox['m_n1_d'].squeeze(dim=2).t() / nodes.mailbox['m_n1_d'].sum(dim=1).squeeze()).t().view(nodes.mailbox['m_n1_d'].shape)

    n2_d = nodes.mailbox['m_n2_d']
    n_l = nodes.mailbox['m_n_l']
    # n_w = nodes.mailbox['m_n_w']
    # n_d = nodes.mailbox['m_n_d']
    return {'n1_hd': n1_hd, 'n1_d': n1_d, 'n2_hd': n2_hd, 'n2_d': n2_d, 'n_l': n_l}
    # return {'n1_h': n1_h, 'n1_hd': n1_hd, 'n2_hd': n2_hd, 'n1_w': n1_w, 'n1_d': n1_d, 'n2_d': n2_d, 'n_w': n_w, 'n_d': n_d}
    # return {'n1_v': n1_v, 'n2_v': n2_v, 'n1_e': n1_e, 'n2_e': n2_e, 'n1_h': n1_h, 'n1_w': n1_w, 'n1_d': n1_d, 'n_w': n_w, 'n_d': n_d}

class NodeApplyModule(nn.Module):
    def __init__(self, k, n, hidden_dim, activation, edge_info='adj_dist'):
        super(NodeApplyModule, self).__init__()
        self.n = n
        self.l1 = nn.Linear(k, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l2_ = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(k + 1, hidden_dim)
        self.l4 = nn.Linear(hidden_dim + 1, hidden_dim)
        self.l5 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.l6 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.edge_info = edge_info
        # self.l4 = nn.Linear(ajr, hidden_dim)
        # self.l5 = nn.Linear(m-1, hidden_dim)

        self.t3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.t3_ = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.t4 = nn.Linear(1, hidden_dim, bias=True)
        self.t4_ = nn.Linear(1, hidden_dim, bias=True)
        self.t5 = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.activation = activation

    def forward(self, node):
        l = node.data['label']  # node label [l]  (bn, k)
        bn = l.shape[0]
        n_l = node.data['n_l']  # reserved adjacent label [l]  (bn, n-1, k)

        n1_d = node.data['n1_d']  # reserved adjacent dist [d]  (bn, n-1, 1)
        n1_hd = node.data['n1_hd']  # reserved adjacent dist [h, d]  (bn, n-1, h)

        n1_h = torch.bmm(n1_hd.transpose(1, 2), n1_d).squeeze(2)  # (bn, h)

        # x = self.activation(self.l4(torch.cat([torch.mean(self.activation(self.l3(torch.cat([n_l, n1_d], dim=2))), dim=1), torch.ones(bn, 1).cuda() * self.n], dim=1)))  #  (bn, n-1, k+1) -> (bn, h)
        #
        # m = self.activation(self.l5(torch.cat([self.activation(self.l2(n1_h)) * (1 / (self.n - 1)), x], dim=1)))
        #
        # h = self.activation(self.l6(torch.cat([node.data['h'], m], dim=1)))

        h = self.activation(self.l1(l)
                            + self.l2(n1_h)
                            + self.t3(torch.sum(self.activation(self.t4(n1_d), inplace=True), dim=1))
                            , inplace=True)

        # h = self.activation(self.l1(l)
        #                     + self.l2(n1_h)
        #                     + self.t3(torch.sum(self.activation(self.t4(n1_d), inplace=True), dim=1))
        #                     , inplace=True)

        return {'h': h}

    def forward2(self, node):
        l = node.data['label']

        n2_d = node.data['n2_d']  # cluster dist

        n2_hd = node.data['n2_hd']  # cluster dist [h, d]

        n2_h = torch.bmm(n2_hd.transpose(1, 2), n2_d).squeeze(2)

        h = self.activation(
            self.l1(l) + self.l2_(n2_h) + self.t3_(torch.sum(self.activation(self.t4_(n2_d), inplace=True), dim=1)),
            inplace=True)

        return {'h': h}

class GCN(nn.Module):
    def __init__(self, k, n, hidden_dim, activation=F.relu, edge_info='adj_dist'):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(k, n, hidden_dim, activation, edge_info=edge_info)
        self.apply_mod2 = NodeApplyModule(k, n, hidden_dim, activation, edge_info='aux')

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(udf_u_mul_e, reduce)
        g.apply_nodes(func=self.apply_mod.forward)
        # g.apply_nodes(func=self.apply_mod.forward2)
        g.ndata.pop('n2_d')
        g.ndata.pop('n2_hd')
        g.ndata.pop('n1_d')
        g.ndata.pop('n1_hd')
        g.ndata.pop('n_l')
        return g.ndata.pop('h')

    def forward2(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(udf_u_mul_e, reduce)
        g.apply_nodes(func=self.apply_mod2)
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
    def __init__(self, k, m, ajr, num_head, hidden_dim, extended_h=True, use_x=False, edge_info='adj_dist', readout='mlp'):
        super(DQNet, self).__init__()
        self.k = k
        self.m = m
        self.n = k * m
        self.num_head = num_head
        self.extended_h = extended_h
        self.hidden_dim = hidden_dim
        if self.extended_h:
            self.hidden_dim += k
            self.true_dim = self.hidden_dim - k
        self.value1 = nn.Linear(num_head*self.hidden_dim, hidden_dim//2)
        self.value2 = nn.Linear(hidden_dim//2, 1)
        self.layers = nn.ModuleList([GCN(k, self.n, hidden_dim, F.relu)])
        self.MHA = MultiHeadedAttention(h=num_head
                           , d_model=num_head*self.hidden_dim
                           , dropout=0.0
                           , activate_linear=True)
        # baseline
        self.t5 = nn.Linear(self.hidden_dim, 1)
        self.t6 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.t66 = nn.Linear(5, self.hidden_dim)
        # # for centroid graph representation
        # self.t5_ = nn.Linear((self.k + 2) * self.hidden_dim, 1)
        self.t6_ = nn.Linear(self.hidden_dim * self.k, self.hidden_dim * self.k)

        self.t7 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.t7_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t7_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.t7_ = nn.Linear(2 * self.true_dim + 2 * self.hidden_dim, self.hidden_dim)
        self.t9 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.t9_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t9_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t9_ = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        # self.t77 = nn.Linear(10, self.hidden_dim)
        self.t8 = nn.Linear(self.hidden_dim + self.k, self.hidden_dim)

        self.h_residual = []
        self.readout = readout
        self.info = ''

        # params for S-eval net
        self.u5 = nn.Linear(self.hidden_dim, 1)
        self.u6 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.u6_ = nn.Linear((self.k+1) * self.hidden_dim + 8, 2*self.hidden_dim)
        self.u9 = nn.Linear(2*self.hidden_dim, self.hidden_dim)


    def forward_nognn(self, g, actions=None, action_type='swap', gnn_step=3, time_aware=False, remain_episode_len=None):
        if isinstance(g, BatchedDGLGraph):
            batch_size = g.batch_size
            num_action = actions.shape[0] // batch_size
        else:
            num_action = actions.shape[0]


        g.ndata['h'] = torch.cat([g.ndata['x'], g.ndata['label']], dim=1)

        # compute group centroid for batch graph g
        # batch_centroid = []
        # for i in range(self.k):
        #     blocks_i = g.ndata['h'][(g.ndata['label'][:, i] > .5).nonzero().squeeze()]
        #     batch_centroid.append(torch.mean(blocks_i.view(batch_size, self.m, self.hidden_dim), axis=1))

        if isinstance(g, BatchedDGLGraph):
            # graph_embedding = self.t6_(torch.cat(batch_centroid, axis=1)).repeat(1, num_action).view(num_action * batch_size, -1) # batch_size * (h * k)
            graph_embedding = self.t66(dgl.mean_nodes(g, 'h')).repeat(1, num_action).view(num_action * batch_size, -1)
        else:
            graph_embedding = self.t66(torch.mean(g.ndata['h'], dim=0)).repeat(num_action, 1)

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
            q_actions = self.t77(torch.cat([g.ndata['h'][actions_[:, 0], :], g.ndata['h'][actions_[:, 1], :]], axis=1))

        S_a_encoding = torch.cat([graph_embedding, q_actions], axis=1)

        Q_sa = self.t5(F.relu(S_a_encoding, inplace=True)).squeeze()
        # Q_sa = self.t5_(F.relu(S_a_encoding)).squeeze()
        # Q_sa = (Q_sa.view(n, n) + Q_sa.view(n, n).t()).view(n**2)

        return S_a_encoding, g.ndata['h'], g.ndata['h'], Q_sa

    def forward_state_eval(self, g, S_enc, gnn_step=3):

        batch_size = g.batch_size
        h = torch.zeros((g.ndata['h'].shape[0], self.hidden_dim - self.k)).cuda()

        for i in range(gnn_step):
            for conv in self.layers:
                h = conv.forward2(g, h)

        g.ndata['h'] = torch.cat([h, g.ndata['label']], dim=1)
        h_new = h

        centroid = [g.ndata['h'][g.ndata['label'][:, ki] == 1].view(batch_size, self.k, -1).sum(dim=1) / self.m for ki in range(self.k)]
        centroid.append(S_enc)
        graph_embedding = F.relu(self.u6_(torch.cat(centroid, dim=1)))  # (b, (k + 1) * h)
        # graph_embedding = self.u6(dgl.mean_nodes(g, 'h'))

        S_encoding = self.u9(torch.cat([graph_embedding], dim=1))

        S_eval = self.u5(F.relu(S_encoding, inplace=True)).squeeze()

        return S_encoding, h_new, g.ndata['h'], S_eval

    def forward_(self, g, actions=None, action_type='swap', gnn_step=3, time_aware=False, remain_episode_len=None, aux_output=False):

        batch_size = g.batch_size
        num_action = actions.shape[0] // batch_size

        # GCN layers
        h = torch.zeros((batch_size * self.n, self.hidden_dim - self.k)).cuda()
        for i in range(gnn_step):
            for conv in self.layers:
                h = conv(g, h)

        g.ndata['h'] = torch.cat([h, g.ndata['label']], dim=1)

        # average readout
        graph_embedding_stem = self.t6(dgl.mean_nodes(g, 'h'))  # (b, h)

        # action indices
        action_mask = torch.tensor(range(0, self.n * batch_size, self.n)) \
            .unsqueeze(1).expand(batch_size, 2) \
            .repeat(1, num_action) \
            .view(num_action * batch_size, -1)

        actions_ = actions + action_mask.cuda()

        g.ndata['h'] = self.t7_1(g.ndata['h'])  # (b*n_a, h)

        Q_sa = self.t5(
            F.relu(
                (self.t9_1(
                            F.relu(graph_embedding_stem, inplace=True)
                        ).view(batch_size, 1, -1)
                    + self.t9_2(
                            F.relu(self.t7_2(F.relu(g.ndata['h'][actions_[:, 0], :] + g.ndata['h'][actions_[:, 1], :], inplace=True)), inplace=True)
                        ).view(batch_size, num_action, -1)
                 ).view(batch_size * num_action, -1), inplace=True)
            ).squeeze()

        # Q_sa = self.t5(
        #     F.relu(
        #         (self.t9_1(
        #                     graph_embedding_stem
        #                 ).view(batch_size, 1, -1)
        #             + self.t9_2(
        #                     self.t7_2(g.ndata['h'][actions_[:, 0], :] + g.ndata['h'][actions_[:, 1], :])
        #                 ).view(batch_size, num_action, -1)
        #          ).view(batch_size * num_action, -1), inplace=True)
        #     ).squeeze()

        g.ndata.pop('h')
        if aux_output:
            return graph_embedding_stem, h, h, Q_sa
        else:
            return graph_embedding_stem, h, h, Q_sa

    def forward(self, g, actions=None, action_type='swap', gnn_step=3, time_aware=False, remain_episode_len=None, aux_output=False):

        if self.readout == 'att':
            return self.forward_MHA(g=g, actions=actions, action_type=action_type, gnn_step=gnn_step, time_aware=time_aware, remain_episode_len=remain_episode_len)

        if isinstance(g, BatchedDGLGraph):
            batch_size = g.batch_size
            num_action = actions.shape[0] // batch_size
            g.ndata['h'] = torch.zeros((g.number_of_nodes(), self.hidden_dim))
        else:
            num_action = actions.shape[0]
            g.ndata['h'] = torch.zeros((g.number_of_nodes(), self.hidden_dim))


        if self.extended_h:
            h = torch.zeros((g.ndata['h'].shape[0], self.hidden_dim - self.k)).cuda()
        else:
            h = torch.zeros((g.ndata['h'].shape[0], self.hidden_dim)).cuda()

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
        # batch_centroid = []
        # for i in range(self.k):
        #     blocks_i = g.ndata['h'][(g.ndata['label'][:, i] > .5).nonzero().squeeze()]
        #     batch_centroid.append(torch.mean(blocks_i.view(batch_size, self.m, self.hidden_dim), axis=1))

        if isinstance(g, BatchedDGLGraph):
            # pass
            # graph_embedding = 0
            # S_a_encoding = 0
            # centroid = [g.ndata['h'][g.ndata['label'][:, ki] == 1].view(batch_size, self.m, -1).sum(dim=1) / self.m for ki in range(self.k)]
            # key = torch.cat(centroid, dim=1).view(batch_size, self.k, -1)  # (b, k, h)
            # key = torch.cat([key, key], dim=2)  # (b, k, 2h)
            # value = key  # (b, k, 2h)
            # graph_embedding_stem = self.t6(dgl.mean_nodes(g, 'h'))  # (b, h)
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
            # g.ndata['h'] = self.t7_1(g.ndata['h'])  # (b*n_a, h)
            # q_actions_stem = self.t7_2(g.ndata['h'][actions_[:, 0], :] + g.ndata['h'][actions_[:, 1], :])  # (b*n_a, h)
            q_actions = self.t7(torch.cat([g.ndata['h'][actions_[:, 0], :], g.ndata['h'][actions_[:, 1], :]], axis=1))

        # Q = torch.cat([g.ndata['h'][actions_[:, 0], :], \
        #                  g.ndata['h'][actions_[:, 1], :]], axis=1).view(batch_size, num_action, -1)
        # # (b, num_action, 2h)
        # graph_embedding = self.MHA(Q, key, value).view(batch_size*num_action, -1)  # (bg, num_action, 1, 2h)

        S_a_encoding = self.t9(torch.cat([graph_embedding, q_actions], axis=1))
        # S_a_encoding = self.t9_(torch.cat([graph_embedding, q_actions], axis=1))
        # S_a_encoding = torch.cat([graph_embedding, q_actions], axis=1)
        # S_a_encoding = (self.t9_1(self.t6(dgl.mean_nodes(g, 'h'))).view(batch_size, 1, -1)
        #                 + self.t9_2(self.t7_2(g.ndata['h'][actions_[:, 0], :] + g.ndata['h'][actions_[:, 1], :])).view(batch_size, num_action, -1)).view(batch_size * num_action, -1)
        Q_sa = self.t5(F.relu(S_a_encoding, inplace=True)).squeeze()
        # Q_sa = self.t5(
        #     F.relu(
        #         (self.t9_1(
        #                     F.relu(self.t6(dgl.mean_nodes(g, 'h')), inplace=True)
        #                 ).view(batch_size, 1, -1)
        #             + self.t9_2(
        #                     F.relu(self.t7_2(g.ndata['h'][actions_[:, 0], :] + g.ndata['h'][actions_[:, 1], :]), inplace=True)
        #                 ).view(batch_size, num_action, -1)
        #          ).view(batch_size * num_action, -1), inplace=True)
        #     ).squeeze()
        # Q_sa = F.relu(Q_sa) - (Q_sa < 0) * 1e-8
        # Q_sa.view(-1, num_action)[:, -1] *= 0
        # Q_sa = self.t5_(F.relu(S_a_encoding)).squeeze()
        # Q_sa = (Q_sa.view(n, n) + Q_sa.view(n, n).t()).view(n**2)

        g.ndata.pop('h')
        if aux_output:
            return graph_embedding, h_new, h, Q_sa
        else:
            return S_a_encoding, h_new, h, Q_sa

    def forward_MHA(self, g, actions=None, action_type='swap', gnn_step=3, time_aware=False, remain_episode_len=None):
        n = self.n
        k = self.k
        m = self.m
        hidden_dim = self.hidden_dim
        num_head = self.num_head

        if isinstance(g, BatchedDGLGraph):
            batch_size = g.batch_size
            num_action = actions.shape[0] // batch_size
            g.ndata['h'] = torch.zeros((self.n * batch_size, self.hidden_dim))
        else:
            num_action = actions.shape[0]
            g.ndata['h'] = torch.zeros((self.n, self.hidden_dim))

        # h = g.ndata['h']
        if self.extended_h:
            h = torch.zeros((g.ndata['h'].shape[0], self.hidden_dim - self.k)).cuda()
        else:
            h = torch.zeros((g.ndata['h'].shape[0], self.hidden_dim)).cuda()

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

        Q = torch.cat([q_h, q_c], axis=2)  # (bg, num_action, 4d)

        S_a_encoding = self.MHA(Q, key, value)# (bg, num_action, 1, 4d)
        # mN = dgl.mean_nodes(g, 'h')
        # PI = self.policy(g.ndata['h'])
        Q_sa = self.value2(F.relu(self.value1(S_a_encoding), inplace=True))
        g.ndata.pop('h')
        return S_a_encoding, h, h, Q_sa.squeeze().view(-1)
