import math
from copy import deepcopy as dc
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, k, hidden_dim, activation=F.relu):
        super(GCN, self).__init__()
        self.l1 = nn.Linear(k, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.activ = activation

    def forward(self, graphs, feature):

        b = graphs.batch_size
        n = graphs.n

        l = graphs.ndata['label']  # node label [l]  (bn, k)

        adjM = graphs.ndata['adj']  # adjacent matrix [d]  (bn, n)

        n1_h = torch.bmm(adjM.view(b, n, n), feature.view(b, n, -1)).view(b * n, -1)  # (bn, h)

        h = self.activ(self.l1(l) + self.l2(n1_h), inplace=True)

        return h

# gnn = GCN(k=3, hidden_dim=16)
# h = torch.zeros(18, 16)
# h = gnn.forward(gg, h)


class DQNet(nn.Module):
    def __init__(self, k, hidden_dim):
        super(DQNet, self).__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([GCN(k, hidden_dim)])
        # baseline
        self.t5 = nn.Linear(self.hidden_dim, 1)
        self.t6 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t7 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.t8 = nn.Linear(self.hidden_dim + self.k, self.hidden_dim)
        self.t9 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

    def forward(self, graphs, actions=None, action_type='swap', gnn_step=3, aux_output=False):

        n = graphs.n
        b = graphs.batch_size
        bn = b * n
        num_action = actions.shape[0] // b

        h = torch.zeros((bn, self.hidden_dim))
        # h[:, 0] += n
        if graphs.in_cuda:
            h = h.cuda()

        for _ in range(gnn_step):
            h = self.layers[0].forward(graphs, h)  # (bn, h)

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

        cluster_embedding = h.view(b, n, -1).mean(dim=1)  # (b, h)
        # cluster_embedding = batch_centroid_H.mean(dim=1)  # (b, h)

        # compute group centroid for batch graph g
        # batch_centroid = []
        # for i in range(self.k):
        #     blocks_i = g.ndata['h'][(g.ndata['label'][:, i] > .5).nonzero().squeeze()]
        #     batch_centroid.append(torch.mean(blocks_i.view(batch_size, self.m, self.hidden_dim), axis=1))

        graph_embedding = self.t6(cluster_embedding).repeat(1, num_action).view(num_action * b, -1)  # (b*num_action, h)



        action_mask = torch.tensor(range(0, bn, n))\
            .unsqueeze(1).expand(b, 2)\
            .repeat(1, num_action)\
            .view(num_action * b, -1)
        # action_cluster_mask = torch.tensor(range(0, batch_size * self.k, self.k))\
        #     .unsqueeze(1).expand(batch_size, 1)\
        #     .repeat(1, num_action)\
        #     .view(num_action * batch_size)
        if graphs.in_cuda:
            action_mask = action_mask.cuda()
        actions_ = actions + action_mask


        # i_cluster, j_cluster = g.ndata['label'][actions_[:, 0], :].nonzero()[:, 1], g.ndata['label'][actions_[:, 1],
        #                                                                             :].nonzero()[:, 1]
        # i_cluster += action_cluster_mask.cuda()
        # j_cluster += action_cluster_mask.cuda()

        if action_type == 'flip':
            q_actions = self.t8(torch.cat([h[actions_[:, 0], :], torch.nn.functional.one_hot(actions[:, 1], self.k).float()], axis=1))
        if action_type == 'swap':
            # q_actions = torch.cat([g.ndata['h'][actions_[:, 0], :], g.ndata['h'][actions_[:, 1], :]], axis=1)
            # g.ndata['h'] = self.t7_1(g.ndata['h'])  # (b*n_a, h)
            # q_actions_stem = self.t7_2(g.ndata['h'][actions_[:, 0], :] + g.ndata['h'][actions_[:, 1], :])  # (b*n_a, h)
            # q_actions = self.h4h(torch.cat([batch_centroid_H.view(batch_size*self.k,-1)[i_cluster, :], batch_centroid_H.view(batch_size*self.k,-1)[j_cluster, :], g.ndata['h'][actions_[:, 0], :], g.ndata['h'][actions_[:, 1], :]], axis=1))

            q_actions = self.t7(torch.cat([h[actions_[:, 0], :], h[actions_[:, 1], :]], axis=1))

        # assign 0 embedding for dummy actions: when num_action > 1, set the dummy actions at the end of the sequence by default
        if num_action > 1 and action_type == 'flip':
            q_actions[range(num_action - 1, b * num_action, num_action), :] *= 0
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

        return S_a_encoding, h, h, Q_sa

# dqn = DQNet(3, 16)
# dqn.forward(gg, a[:10, :])
