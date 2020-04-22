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
        if isinstance(m, list):
            self.n = sum(m)
            assert len(m) == k
            self.cut = 'unequal'
            self.init_label = []
            for i in range(k):
                self.init_label.extend([i]*m[i])
        else:
            self.cut = 'equal'
            self.n = k * m
        self.ajr = ajr
        self.nonzero_idx = [i for i in range(self.n**2) if i % (self.n+1) != 0]
        self.src = [i // self.n for i in self.nonzero_idx]
        self.dst = [i % self.n for i in self.nonzero_idx]
        self.adj_mask = torch.tensor(range(0, self.n ** 2, self.n)).unsqueeze(1).expand(self.n, ajr + 1)
        self.style = style

    def generate_G(self, x=None, label=None, hidden_dim=16):

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
            if self.cut == 'equal':
                label = torch.tensor(range(k)).unsqueeze(1).expand(k, m).flatten()
                label = label[torch.randperm(n)]
            else:
                label = torch.tensor(self.init_label)[torch.randperm(n)]
        else:
            label = torch.tensor(label)

        g.ndata['label'] = torch.nn.functional.one_hot(label, k).float()

        _, neighbor_idx, dist_matrix = dgl.transform.knn_graph(g.ndata['x'], self.ajr + 1, extend_info=True)

        g.add_edges(self.src, self.dst)
        g.ndata['adj'] = torch.sqrt(dist_matrix[0])
        g.edata['d'] = g.ndata['adj'].view(-1, 1)[self.nonzero_idx, :]

        adjacent_matrix = torch.zeros((n * n, 1))
        adjacent_matrix[neighbor_idx[0] + self.adj_mask] = 1

        group_matrix = torch.mm(g.ndata['label'], g.ndata['label'].t()).view(-1, 1)

        g.edata['e_type'] = torch.cat([adjacent_matrix[self.nonzero_idx, :], group_matrix[self.nonzero_idx, :]], dim=1)

        # init node embedding h
        g.ndata['h'] = torch.zeros((g.number_of_nodes(), hidden_dim))

        return g

    def generate_batch_G(self, target_bg=None, x=None, batch_size=1, style=None, seed=None):

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

        # calculate edges
        if target_bg is not None:
            # permute the dist matrix
            # TODO: add ndata['adj']
            bg.edata['d'] *= F.relu(torch.ones(bg.edata['d'].shape).cuda() + 0.1 * torch.randn(bg.edata['d'].shape).cuda())
        else:
            if style.startswith('er') or style.startswith('ba'):
                # TODO: add ndata['adj']
                bg.edata['d'] = adj_matrices.view(batch_size, -1, 1)[:, self.nonzero_idx, :].view(-1, 1)
            else:
                _, neighbor_idx, square_dist_matrix = dgl.transform.knn_graph(bg.ndata['x'].view(batch_size, n, -1), ajr + 1, extend_info=True)
                square_dist_matrix = F.relu(square_dist_matrix, inplace=True)  # numerical error could result in NaN in sqrt. value
                bg.ndata['adj'] = torch.sqrt(square_dist_matrix).view(bg.number_of_nodes(), -1)
                # scale d (maintain avg=0.5):
                if style != 'plain':
                    bg.ndata['adj'] /= (bg.ndata['adj'].sum() / (bg.ndata['adj'].shape[0]**2) / 0.5)
                bg.edata['d'] = bg.ndata['adj'].view(batch_size, -1, 1)[:, self.nonzero_idx, :].view(-1, 1)

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

# G=GraphGenerator(3,3,8, style='plain')
# g=G.generate_batch_G(batch_size=2)

def peek_greedy_reward(states, actions=None, action_type='swap'):
    # g = problem.g
    # states = dgl.batch([g,g])
    # actions = problem.get_legal_actions(state=states)
    if isinstance(states, BatchedDGLGraph):
        batch_size = states.batch_size
    else:
        batch_size = 1
    n = states.ndata['label'].shape[0] // batch_size
    bn = batch_size * n

    if actions is None:
        mask = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
                         states.ndata['label'].view(batch_size, n, -1).transpose(1, 2))
        legal_actions = torch.triu(1 - mask).nonzero()[:, 1:3]  # tensor (270, 2)
        legal_actions = legal_actions.reshape(batch_size, -1, 2)
        actions = torch.cat([legal_actions, (legal_actions[:, 0] * 0).unsqueeze(1)], dim=1).view(-1, 2)
        group_matrix = mask.view(bn, n)
    else:
        group_matrix = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
                                 states.ndata['label'].view(batch_size, n, -1).transpose(1, 2)).view(bn, n)
    num_action = actions.shape[0] // batch_size

    action_mask = torch.tensor(range(0, bn, n)).unsqueeze(1).expand(batch_size, 2).repeat(1, num_action).view(
        num_action * batch_size, -1).cuda()
    actions_ = actions + action_mask
    #  (b, n, n)
    #  (b * num_action, n)
    rewards = (states.ndata['adj'][actions_[:, 0], :] * (
                group_matrix[actions_[:, 0], :] - group_matrix[actions_[:, 1], :])).sum(dim=1) \
              + (states.ndata['adj'][actions_[:, 1], :] * (
                group_matrix[actions_[:, 1], :] - group_matrix[actions_[:, 0], :])).sum(dim=1) \
              + 2 * states.ndata['adj'][actions_[:, 0], actions[:, 1]]

    return rewards

# peek_greedy_reward(states=dgl.batch([g,g,g]))

class KCut_DGL:

    def __init__(self, k, m, adjacent_reserve, hidden_dim, mode='complete', x=None, label=None, graph_style='plain', sample_episode=10):
        self.graph_generator = GraphGenerator(k=k, m=m, ajr=adjacent_reserve, style=graph_style)
        self.g = self.graph_generator.generate_G(x=x, label=label, hidden_dim=hidden_dim)
        self.k = k  # num of clusters
        # self.m = m
        if isinstance(m, list):
            self.N = sum(m)
            assert len(m) == k
        else:
            self.N = k * m
        self.adjacent_reserve = adjacent_reserve
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.x = x
        self.label = label
        self.S = self.calc_S()
        self.sample_episode = sample_episode
        self.gen_step_batch_mask()

    def gen_step_batch_mask(self):
        # assert(sample_episode == states.batch_size) in step_batch
        self.mask1 = torch.tensor(range(0, self.N * self.sample_episode, self.N)).cuda()
        # self.mask2 = torch.tensor(range(0, self.N * self.sample_episode, self.N)).repeat(self.m, 1).t().flatten().cuda()

    def calc_S(self, g=None):
        if g is None:
            g = self.g

        S = g.edata['e_type'][:, 1] * g.edata['d'][:, 0]
        if self.mode != 'complete':
            S *= g.edata['e_type'][:, 0]
        return torch.sum(S) / 2

    def calc_batchS(self, bg):

        S = bg.edata['e_type'][:, 1] * bg.edata['d'][:, 0]
        if self.mode != 'complete':
            S *= bg.edata['e_type'][:, 0]
        return S.view(bg.batch_size, -1).sum(dim=1) / 2


    def gen_batch_graph(self, x=None, batch_size=1, style=None, seed=None):
        return self.graph_generator.generate_batch_G(x=x, batch_size=batch_size, style=style, seed=seed)

    def gen_target_batch_graph(self, target_bg=None, x=None, batch_size=1, style=None):
        return self.graph_generator.generate_batch_G(target_bg=target_bg, x=x, batch_size=batch_size, style=style)

    def reset(self, compute_S=True):
        self.g = self.graph_generator.generate_G(x=self.x, label=self.label)
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
        n = label.shape[0]
        if rewire_edges:
            nonzero_idx = [i for i in range(n ** 2) if i % (n + 1) != 0]
            g.edata['e_type'][:, 1] = torch.mm(g.ndata['label'], g.ndata['label'].t()).view(-1)[nonzero_idx]

        if g is None and calc_S:
            self.S = self.calc_S()
        return g

    def set_batch_label(self, bg, label):
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        bg.ndata['label'] = torch.nn.functional.one_hot(label, self.k).float().cuda()
        # rewire edges
        n = bg.ndata['label'].shape[0] // bg.batch_size
        nonzero_idx = [i for i in range(n ** 2) if i % (n + 1) != 0]
        bg.edata['e_type'][:, 1] = torch.bmm(bg.ndata['label'].view(bg.batch_size, n, -1)
                                             , bg.ndata['label'].view(bg.batch_size, n, -1).transpose(1, 2)) \
                                       .view(bg.batch_size, -1)[:, nonzero_idx].view(-1)

    def get_legal_actions(self, state=None, action_type='swap', action_dropout=1.0, pause_action=True):

        if state is None:
            state = self.g

        if action_type == 'flip':
            if isinstance(state, BatchedDGLGraph):
                legal_actions = torch.nonzero(1 - state.ndata['label'])
                num_actions = legal_actions.shape[0] // state.batch_size
                mask = torch.tensor(range(0, self.N*state.batch_size, self.N)).repeat(num_actions).view(-1, state.batch_size).t().flatten().cuda()
                legal_actions[:, 0] -= mask

                if action_dropout < 1.0:
                    maintain_actions = int(num_actions * action_dropout)
                    maintain = [np.random.choice(range(_ * num_actions, (_ + 1) * num_actions), maintain_actions, replace=False) for _ in range(state.batch_size)]
                    legal_actions = legal_actions[torch.tensor(maintain).flatten(), :]
                if pause_action:
                    legal_actions = legal_actions.reshape(state.batch_size, -1, 2)
                    legal_actions = torch.cat([legal_actions, (legal_actions[:, 0] * 0 ).unsqueeze(1)], dim=1).view(-1, 2)

            else:
                legal_actions = torch.nonzero(1 - state.ndata['label'])
                # action_dropout:
                if action_dropout < 1.0:
                    num_actions = legal_actions.shape[0]
                    maintain_actions = int(num_actions * action_dropout)
                    maintain = np.random.choice(range(num_actions), maintain_actions, replace=False)
                    legal_actions = legal_actions[torch.tensor(maintain).flatten(), :]
                if pause_action:
                    # add a special pause action (0, 0)
                    legal_actions = torch.cat([legal_actions, (legal_actions[0] * 0).unsqueeze(0)], dim=0)

        if action_type == 'swap':
            if isinstance(state, BatchedDGLGraph):
                n = state.number_of_nodes() // state.batch_size
                mask = torch.bmm(state.ndata['label'].view(state.batch_size, n, -1),
                                 state.ndata['label'].view(state.batch_size, n, -1).transpose(1, 2))
                legal_actions = torch.triu(1 - mask).nonzero()[:, 1:3]  # tensor (270, 2)

                # action_dropout:g

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

    def step(self, action, action_type='swap', state=None, rewire_edges=True):

        if state is None:
            state = self.g

        if action_type == 'swap':
            i, j = action
            if i + j == 0:
                return state, torch.tensor(.0)

            old_S = self.calc_S(g=state)
            # swap two nodes
            tmp = dc(state.nodes[i].data['label'])
            state.nodes[i].data['label'] = state.nodes[j].data['label']
            state.nodes[j].data['label'] = tmp

            # rewire edges
            if rewire_edges:
                state.edata['e_type'][:, 1] = torch.mm(state.ndata['label'], state.ndata['label'].t()).view(-1)[
                    self.graph_generator.nonzero_idx]

        elif action_type == 'flip':
            i = action[0]
            target_label = action[1]
            i_label = state.nodes[i].data['label'].argmax().item()

            if i_label == target_label or i + target_label < 0:
                return state, torch.tensor(.0)

            old_S = self.calc_S(g=state)

            # flip node
            state.nodes[i].data['label'] = torch.nn.functional.one_hot(torch.tensor([target_label]), self.k).float()

            # rewire edges
            if rewire_edges:
                state.edata['e_type'][:, 1] = torch.mm(state.ndata['label'], state.ndata['label'].t()).view(-1)[
                    self.graph_generator.nonzero_idx]

        new_S = self.calc_S(g=state)
        reward = old_S - new_S

        if state is None:
            self.S -= reward

        return state, reward

    def step_batch(self, states, action, action_type='swap', return_sub_reward=False):
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

        old_S = self.calc_batchS(bg=states)

        if action_type == 'swap':
            # swap two sets of nodes
            tmp = dc(states.ndata['label'][ii + self.mask1])
            states.ndata['label'][ii + self.mask1] = states.ndata['label'][jj + self.mask1]
            states.ndata['label'][jj + self.mask1] = tmp
        else:
            # flip nodes
            states.ndata['label'][ii + self.mask1] = torch.nn.functional.one_hot(jj, self.k).float()
        # rewire edges
        states.edata['e_type'][:, 1] = torch.bmm(states.ndata['label'].view(batch_size, n, -1),
                                                 states.ndata['label'].view(batch_size, n, -1).transpose(1, 2)) \
                                           .view(batch_size, -1)[:, self.graph_generator.nonzero_idx].view(-1)

        # compute new S
        new_S = self.calc_batchS(bg=states)

        rewards = old_S - new_S

        # if return_sub_reward:
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

        return states, rewards


def udf_u_mul_e(edges):
    # a= edges.data['d'] * edges.data['e_type'][:, 0].unsqueeze(1)
    # print(a.view(15,14,1))
    # print('h shape:', edges.src['h'].shape)
    # print('d shape:', edges.data['d'].shape)
    return {
        'm_n1_h': edges.src['h'] * edges.data['e_type'][:, 0].unsqueeze(1)
        , 'm_n2_h': edges.src['h'] * edges.data['e_type'][:, 1].unsqueeze(1)
        , 'm_n3_h': edges.src['h'] * (edges.data['e_type'][:, 0]-edges.data['e_type'][:, 1]).unsqueeze(1)
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
        , 'm_n3_d': edges.data['d'] * (edges.data['e_type'][:, 0]-edges.data['e_type'][:, 1]).unsqueeze(1)
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
    n3_hd = nodes.mailbox['m_n3_h']
    # nodes.mailbox['m_n1_d']: of size (n*b, n-1, 1)
    n1_d = nodes.mailbox['m_n1_d']
    # n1_d = (nodes.mailbox['m_n1_d'].squeeze(dim=2).t() / nodes.mailbox['m_n1_d'].sum(dim=1).squeeze()).t().view(nodes.mailbox['m_n1_d'].shape)

    n2_d = nodes.mailbox['m_n2_d']
    n3_d = nodes.mailbox['m_n3_d']
    n_l = nodes.mailbox['m_n_l']
    # n_w = nodes.mailbox['m_n_w']
    # n_d = nodes.mailbox['m_n_d']
    return {'n1_hd': n1_hd, 'n1_d': n1_d, 'n2_hd': n2_hd, 'n2_d': n2_d, 'n3_hd': n3_hd, 'n3_d': n3_d, 'n_l': n_l}
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

        # n2_d = node.data['n2_d']  # reserved adjacent dist [d]  (bn, n-1, 1)
        # n2_hd = node.data['n2_hd']  # reserved adjacent dist [h, d]  (bn, n-1, h)
        # n2_h = torch.bmm(n2_hd.transpose(1, 2), n2_d).squeeze(2)  # (bn, h)

        # n3_d = node.data['n3_d']  # reserved adjacent dist [d]  (bn, n-1, 1)
        # n3_hd = node.data['n3_hd']  # reserved adjacent dist [h, d]  (bn, n-1, h)
        # n3_h = torch.bmm(n3_hd.transpose(1, 2), n3_d).squeeze(2)  # (bn, h)

        h = self.activation(self.l1(l)
                            + self.l2(n1_h)
                            + self.t3(torch.sum(self.activation(self.t4(n1_d), inplace=True), dim=1))
                            , inplace=True)

        # h = self.activation(self.l2(n2_h), inplace=True)

        return {'h': h}

    def forward2(self, node):

        l = node.data['label']

        n3_d = node.data['n3_d']  # cluster dist

        n3_hd = node.data['n3_hd']  # cluster dist [h, d]

        n3_h = torch.bmm(n3_hd.transpose(1, 2), n3_d).squeeze(2)

        # h = self.activation(self.l1(l) + self.l2_(n2_h) + self.t3_(torch.sum(self.activation(self.t4_(n2_d), inplace=True), dim=1)), inplace=True)

        h = self.activation(self.l2_(n3_h), inplace=True)

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
        g.ndata.pop('n3_d')
        g.ndata.pop('n3_hd')
        g.ndata.pop('n2_d')
        g.ndata.pop('n2_hd')
        g.ndata.pop('n1_d')
        g.ndata.pop('n1_hd')
        g.ndata.pop('n_l')
        return g.ndata.pop('h')

    def forward2(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(udf_u_mul_e, reduce)
        g.apply_nodes(func=self.apply_mod.forward2)
        # g.apply_nodes(func=self.apply_mod.forward2)
        g.ndata.pop('n3_d')
        g.ndata.pop('n3_hd')
        g.ndata.pop('n2_d')
        g.ndata.pop('n2_hd')
        g.ndata.pop('n1_d')
        g.ndata.pop('n1_hd')
        g.ndata.pop('n_l')
        return g.ndata.pop('h')

# gnn = GCN(k=3, hidden_dim=16)
# problem = KCut_DGL(k=3, m=3, adjacent_reserve=8, hidden_dim=16)
# g = problem.g
# g.ndata['label'] = torch.tensor([
# [0., 1., 1., 1., 0., 0., 0., 0., 0.],
# [0., 0., 0., 0., 1., 0., 1., 1., 0.],
# [1., 0., 0., 0., 0., 1., 0., 0., 1.]]).t()
# nonzero_idx = [i for i in range(9 ** 2) if i % (9 + 1) != 0]
# g.edata['e_type'][:, 1] = torch.mm(g.ndata['label'], g.ndata['label'].t()).view(-1)[nonzero_idx]
# h1, h2 = gnn(g, torch.zeros(9,16), torch.zeros(9,16))
# h1, h2 = gnn(g, h1, h2)
# h1, h2 = gnn(g, h1, h2)
# h2

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


class DQNet(nn.Module):
    # TODO k, m, ajr should be excluded::no generalization
    def __init__(self, k, n, num_head, hidden_dim, edge_info='adj_dist', readout='mlp'):
        super(DQNet, self).__init__()
        self.k = k
        self.n = n
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.value1 = nn.Linear(num_head*self.hidden_dim, hidden_dim//2)
        self.value2 = nn.Linear(hidden_dim//2, 1)
        self.layers = nn.ModuleList([GCN(k, n, hidden_dim, F.relu, edge_info=edge_info)])
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
        self.c1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.hh1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h4h = nn.Linear(4 * self.hidden_dim, self.hidden_dim)

        self.h_residual = []
        self.readout = readout
        self.info = ''

        # params for S-eval net
        self.u5 = nn.Linear(self.hidden_dim, 1)
        self.u6 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.u6_ = nn.Linear((self.k+1) * self.hidden_dim + 8, 2*self.hidden_dim)
        self.u9 = nn.Linear(2*self.hidden_dim, self.hidden_dim)

    def forward_state_eval(self, g, S_enc, gnn_step=3):

        batch_size = g.batch_size
        h = torch.zeros((g.ndata['h'].shape[0], self.hidden_dim - self.k)).cuda()

        for i in range(gnn_step):
            for conv in self.layers:
                h = conv.forward2(g, h)

        g.ndata['h'] = torch.cat([h, g.ndata['label']], dim=1)
        h_new = h

        self_m = 3
        centroid = [g.ndata['h'][g.ndata['label'][:, ki] == 1].view(batch_size, self.k, -1).sum(dim=1) / self_m for ki in range(self.k)]
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

        g.ndata['h'] = torch.cat([h, g.ndata['label']*0], dim=1)

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

    def forward(self, g, actions=None, action_type='swap', gnn_step=3, aux_output=False):

        if self.readout == 'att':
            return self.forward_MHA(g=g, actions=actions, action_type=action_type, gnn_step=gnn_step, time_aware=time_aware, remain_episode_len=remain_episode_len)
        bn = g.number_of_nodes()

        if isinstance(g, BatchedDGLGraph):
            batch_size = g.batch_size
            n = bn // batch_size
            num_action = actions.shape[0] // batch_size
        else:
            num_action = actions.shape[0]
            n = bn

        # all_dist = g.edata['d'] * g.edata['e_type'][:, 0:1]  # (b * n * (n-1), 1)
        # group_dist = g.edata['d'] * g.edata['e_type'][:, 1:2]  # (b * n * (n-1), 1)
        # all_dist = all_dist.view(bn, n - 1).sum(dim=1).unsqueeze(1)
        # group_dist = group_dist.view(bn, n - 1).sum(dim=1).unsqueeze(1)
        # h = torch.cat([torch.cat([all_dist, group_dist], dim=1), torch.zeros((bn, self.hidden_dim - 2)).cuda()], dim=1)
        # h2 = torch.cat([torch.cat([all_dist, group_dist], dim=1), torch.zeros((bn, self.hidden_dim - 2)).cuda()], dim=1)
        h = torch.zeros((bn, self.hidden_dim)).cuda()
        # h[:, 0] += n
        h = self.layers[0].forward(g, h)
        # h = self.layers[0].forward2(g, h)
        h = self.layers[0].forward(g, h)
        # h = self.layers[0].forward2(g, h)
        # h = self.layers[0].forward2(g, h)
        h = self.layers[0].forward(g, h)

        g.ndata['h'] = h  # (bn, h)
        # g.ndata['adj']  # (bn, n)

        # batch_centroid_H = torch.bmm(g.ndata['label'].view(batch_size, n, -1).transpose(1, 2)
        #                             , g.ndata['h'].view(batch_size, n, -1))  # (b, k, h)
        # batch_centroid_adjM = torch.bmm(torch.bmm(g.ndata['label'].view(batch_size, n, -1).transpose(1, 2)
        #                             , g.ndata['adj'].view(batch_size, n, -1)), g.ndata['label'].view(batch_size, n, -1))  # (b, k, k)
        #
        # for _ in range(2):
        #     batch_centroid_H = F.relu(self.hh1(torch.bmm(batch_centroid_adjM, batch_centroid_H)))  # (b, k, h)

        # split_bg = g.ndata['h'].unsqueeze(2).repeat(1, 1, self.k) * g.ndata['label'].view(bn, 1, self.k)  # (bn, h, k)
        # batch_centroids = split_bg.view(batch_size, n, -1, self.k).sum(dim=1).transpose(1, 2)#.reshape(batch_size*self.k, -1)  # (b, k, h)
        # cluster_embedding = F.relu(self.c1(batch_centroids)).mean(dim=1)  # (b, h)


        if isinstance(g, BatchedDGLGraph):
            cluster_embedding = dgl.mean_nodes(g, 'h')
            # cluster_embedding = batch_centroid_H.mean(dim=1)  # (b, h)
        else:
            cluster_embedding = torch.mean(g.ndata['h'], dim=0)
            # cluster_embedding = batch_centroid_H.mean(dim=1)  # (b, h)


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
            graph_embedding = self.t6(cluster_embedding).repeat(1, num_action).view(num_action * batch_size, -1)
        else:
            graph_embedding = self.t6(cluster_embedding).repeat(num_action, 1)

        if isinstance(g, BatchedDGLGraph):
            action_mask = torch.tensor(range(0, bn, n))\
                .unsqueeze(1).expand(batch_size, 2)\
                .repeat(1, num_action)\
                .view(num_action * batch_size, -1)
            # action_cluster_mask = torch.tensor(range(0, batch_size * self.k, self.k))\
            #     .unsqueeze(1).expand(batch_size, 1)\
            #     .repeat(1, num_action)\
            #     .view(num_action * batch_size)

            actions_ = actions + action_mask.cuda()
        else:
            actions_ = actions

        # i_cluster, j_cluster = g.ndata['label'][actions_[:, 0], :].nonzero()[:, 1], g.ndata['label'][actions_[:, 1],
        #                                                                             :].nonzero()[:, 1]
        # i_cluster += action_cluster_mask.cuda()
        # j_cluster += action_cluster_mask.cuda()

        if action_type == 'flip':
            q_actions = self.t8(torch.cat([g.ndata['h'][actions_[:, 0], :], torch.nn.functional.one_hot(actions[:, 1], self.k).float()], axis=1))
        if action_type == 'swap':
            # q_actions = torch.cat([g.ndata['h'][actions_[:, 0], :], g.ndata['h'][actions_[:, 1], :]], axis=1)
            # g.ndata['h'] = self.t7_1(g.ndata['h'])  # (b*n_a, h)
            # q_actions_stem = self.t7_2(g.ndata['h'][actions_[:, 0], :] + g.ndata['h'][actions_[:, 1], :])  # (b*n_a, h)
            # q_actions = self.h4h(torch.cat([batch_centroid_H.view(batch_size*self.k,-1)[i_cluster, :], batch_centroid_H.view(batch_size*self.k,-1)[j_cluster, :], g.ndata['h'][actions_[:, 0], :], g.ndata['h'][actions_[:, 1], :]], axis=1))

            q_actions = self.t7(torch.cat([g.ndata['h'][actions_[:, 0], :], g.ndata['h'][actions_[:, 1], :]], axis=1))

        # assign 0 embedding for dummy actions: when num_action > 1, set the dummy actions at the end of the sequence by default
        if num_action > 1 and action_type=='flip':
            q_actions[range(num_action - 1, batch_size * num_action, num_action), :] *= 0
        # Q = torch.cat([g.ndata['h'][actions_[:, 0], :], \
        #                  g.ndata['h'][actions_[:, 1], :]], axis=1).view(batch_size, num_action, -1)
        # # (b, num_action, 2h)
        # graph_embedding = self.MHA(Q, key, value).view(batch_size*num_action, -1)  # (bg, num_action, 1, 2h)

        S_a_encoding = self.t9(torch.cat([graph_embedding, q_actions], axis=1))
        # S_a_encoding = self.t9_(torch.cat([graph_embedding, q_actions], axis=1))
        # S_a_encoding = torch.cat([graph_embedding, q_actions], axis=1)
        # S_a_encoding = (self.t9_1(self.t6(dgl.mean_nodes(g, 'h'))).view(batch_size, 1, -1)
        #                 + self.t9_2(self.t7_2(g.ndata['h'][actions_[:, 0], :] + g.ndata['h'][actions_[:, 1], :])).view(batch_size, num_action, -1)).view(batch_size * num_action, -1)


        # immediate_rewards = peek_greedy_reward(states=g, actions=actions).unsqueeze(1)

        # print('immediate rewards:', immediate_rewards.shape)
        # print('S_a_encoding', F.relu(S_a_encoding, inplace=True).shape)

        Q_sa = self.t5( torch.cat([F.relu(S_a_encoding, inplace=True)], dim=1) ).squeeze()
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
            return graph_embedding, h, h, Q_sa
        else:
            return S_a_encoding, h, h, Q_sa

    def forward_MHA(self, g, actions=None, action_type='swap', gnn_step=3, time_aware=False, remain_episode_len=None):
        n = self.n
        k = self.k
        hidden_dim = self.hidden_dim
        num_head = self.num_head

        if isinstance(g, BatchedDGLGraph):
            batch_size = g.batch_size
            num_action = actions.shape[0] // batch_size
            g.ndata['h'] = torch.zeros((self.n * batch_size, self.hidden_dim))
        else:
            num_action = actions.shape[0]
            g.ndata['h'] = torch.zeros((self.n, self.hidden_dim))

        h = torch.zeros((g.ndata['h'].shape[0], self.hidden_dim - self.k)).cuda()

        for i in range(gnn_step):
            for conv in self.layers:
                h = conv(g, h)

        g.ndata['h'] = torch.cat([h, g.ndata['label']], dim=1)



        h_new = h
        # pe = PositionalEncoding(h.shape[1], dropout=0, max_len=max_step)
        # g.ndata['h'] = torch.cat([pe(h, remain_step), g.ndata['x'], g.ndata['label']], dim=1)
        # g.ndata['h'] = torch.cat([h, g.ndata['x'], g.ndata['label']], dim=1)
        # g.ndata['h'] = h

        # compute centroid embedding c_i
        # gc_x = torch.mm(g.ndata['x'].t(), g.ndata['label']) / m
        # gc_h = torch.mm(g.ndata['h'].t(), g.ndata['label']) / m
        self_m = 3
        gc_h = torch.bmm(g.ndata['x'].view(batch_size, n, -1).transpose(1, 2), \
                         g.ndata['label'].view(batch_size, n, -1)) / self_m
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
