# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:39 2018

@author: fy4bc
"""
import torch
import numpy as np
import dgl
import torch.nn.functional as F
from copy import deepcopy as dc
import pickle
import os
from test import *
from log_utils import mean_val, logger


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def to_cuda(G_):
    G = dc(G_)
    G.ndata['x'] = G.ndata['x'].cuda()
    G.ndata['label'] = G.ndata['label'].cuda()
    G.ndata['h'] = G.ndata['h'].cuda()
    G.edata['d'] = G.edata['d'].cuda()
    G.edata['w'] = G.edata['w'].cuda()
    G.edata['e_type'] = G.edata['e_type'].cuda()
    return G


class EpisodeHistory:
    def __init__(self, g, max_episode_len):
        self.init_state = g
        self.n = g.number_of_nodes()
        self.max_episode_len = max_episode_len
        self.episode_len = 0
        self.action_seq = []
        self.reward_seq = []
        self.label_perm = torch.tensor(range(self.n)).unsqueeze(0)

    def perm_label(self, label, action):
        label = dc(label)
        tmp = dc(label[action[0]])
        label[action[0]] = label[action[1]]
        label[action[1]] = tmp
        return label.unsqueeze(0)

    def write(self, action, reward):
        self.action_seq.append(action)
        self.reward_seq.append(reward)
        self.label_perm = torch.cat([self.label_perm, self.perm_label(self.label_perm[-1, :], action)], dim=0)
        self.episode_len += 1

    def wrap(self):
        self.reward_seq = torch.tensor(self.reward_seq)
        self.label_perm = self.label_perm.long()


def weight_monitor(model, model_target):
    gnn_0 = torch.mean(model.layers[0].apply_mod.l0.weight).item(), torch.std(model.layers[0].apply_mod.l0.weight).item()
    gnn_1 = torch.mean(model.layers[0].apply_mod.l1.weight).item(), torch.std(model.layers[0].apply_mod.l1.weight).item()
    gnn_2 = torch.mean(model.layers[0].apply_mod.l2.weight).item(), torch.std(model.layers[0].apply_mod.l2.weight).item()
    gnn_3 = torch.mean(model.layers[0].apply_mod.l3.weight).item(), torch.std(model.layers[0].apply_mod.l3.weight).item()
    gnn_4 = torch.mean(model.layers[0].apply_mod.l4.weight).item(), torch.std(model.layers[0].apply_mod.l4.weight).item()
    gnn_5 = torch.mean(model.layers[0].apply_mod.l5.weight).item(), torch.std(model.layers[0].apply_mod.l5.weight).item()
    attn = [(torch.mean(model.MHA.linears[i].weight).item(), torch.std(model.MHA.linears[i].weight).item()) for i in range(4)]
    q_net_1 = torch.mean(model.value1.weight).item(), torch.std(model.value1.weight).item()
    q_net_2 = torch.mean(model.value2.weight).item(), torch.std(model.value2.weight).item()
    gnn_diff0 = torch.norm(model.layers[0].apply_mod.l0.weight - model_target.layers[0].apply_mod.l0.weight).item()
    gnn_diff1 = torch.norm(model.layers[0].apply_mod.l1.weight - model_target.layers[0].apply_mod.l1.weight).item()
    gnn_diff2 = torch.norm(model.layers[0].apply_mod.l2.weight - model_target.layers[0].apply_mod.l2.weight).item()
    gnn_diff3 = torch.norm(model.layers[0].apply_mod.l3.weight - model_target.layers[0].apply_mod.l3.weight).item()
    gnn_diff4 = torch.norm(model.layers[0].apply_mod.l4.weight - model_target.layers[0].apply_mod.l4.weight).item()
    gnn_diff5 = torch.norm(model.layers[0].apply_mod.l5.weight - model_target.layers[0].apply_mod.l5.weight).item()
    attn_diff0 = torch.norm(model.MHA.linears[0].weight - model_target.MHA.linears[0].weight).item()
    attn_diff1 = torch.norm(model.MHA.linears[1].weight - model_target.MHA.linears[1].weight).item()
    attn_diff2 = torch.norm(model.MHA.linears[2].weight - model_target.MHA.linears[2].weight).item()
    attn_diff3 = torch.norm(model.MHA.linears[3].weight - model_target.MHA.linears[3].weight).item()
    q_net_diff1 = torch.norm(model.value1.weight - model_target.value1.weight).item()
    q_net_diff2 = torch.norm(model.value2.weight - model_target.value2.weight).item()

    return {'gnn0':gnn_0, 'gnn1':gnn_1, 'gnn2':gnn_2, 'gnn3':gnn_3, 'gnn4':gnn_4, 'gnn5':gnn_5,
            'attn':attn, 'q_net1':q_net_1, 'q_net2':q_net_2, 'gnn_diff0':gnn_diff0,
            'gnn_diff1':gnn_diff1, 'gnn_diff2':gnn_diff2, 'gnn_diff3':gnn_diff3, 'gnn_diff4':gnn_diff4, 'gnn_diff5':gnn_diff5,
            'attn_diff0':attn_diff0, 'attn_diff1':attn_diff1, 'attn_diff2':attn_diff2, 'attn_diff3':attn_diff3,
            'q_net_diff1':q_net_diff1, 'q_net_diff2':q_net_diff2}


class DQN:
    def __init__(self, problem, gamma=1.0, eps=0.1, lr=1e-4, replay_buffer_max_size=10, cuda_flag=True):

        self.problem = problem
        self.G = problem.g  # the graph
        self.k = problem.k  # num of clusters
        self.m = problem.m  # num of nodes in cluster
        self.ajr = problem.adjacent_reserve  # degree of node in graph
        self.hidden_dim = problem.hidden_dim  # hidden dimension for node representation
        self.n = self.k * self.m  # num of nodes
        self.eps = eps  # constant for exploration in dqn
        if cuda_flag:
            self.model = DQNet(k=self.k, m=self.m, ajr=self.ajr, num_head=4, hidden_dim=self.hidden_dim).cuda()
        else:
            self.model = DQNet(k=self.k, m=self.m, ajr=self.ajr, num_head=4, hidden_dim=self.hidden_dim)
        self.model_target = dc(self.model)
        self.gamma = gamma  # reward decay const
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.experience_replay_buffer = []
        self.replay_buffer_max_size = replay_buffer_max_size
        self.cuda = cuda_flag
        self.log = logger()
        self.Q_err = 0  # Q error
        self.log.add_log('tot_return')
        self.log.add_log('Q_error')
        self.log.add_log('entropy')

    def run_episode(self, gnn_step=10, episode_len=50):
        sum_r = 0
        state = self.problem.reset()
        t = 0

        ep = EpisodeHistory(state, episode_len)

        while t < episode_len:
            if self.cuda:
                G = to_cuda(state)
            else:
                G = dc(state)

            S_a_encoding, h1, h2, Q_sa = self.model(G, gnn_step=gnn_step, max_step=episode_len, remain_step=episode_len-1-t)

            # record
            h_support1 = h1.nonzero().shape[0]
            h_support2 = h2.nonzero().shape[0]
            h_mean = h1.sum() / h_support1
            h_residual = self.model.h_residual
            q_mean = Q_sa.mean()
            q_var = Q_sa.std()
            # model weight

            if t % episode_len in (0, episode_len//2, episode_len-1):
                print('\nh-nonzero entry: %.0f, %.0f'%(h_support1, h_support2))
                print('h-mean: %.5f'%h_mean.item())
                print('h-std: ', ['%.2f'%x.item() for x in h_residual])
                print('q value-mean: %.5f'%q_mean.item())
                print('q value-std: %.5f'%q_var.item())
                print(weight_monitor(self.model, self.model_target))

            # epsilon greedy strategy
            if torch.rand(1) > self.eps:
                best_action = Q_sa.argmax()
            else:
                best_action = torch.randint(high=self.n**2, size=(1,)).squeeze()
            swap_i, swap_j = best_action/self.n, best_action-best_action/self.n*self.n

            # TODO: Re-design reward signal? What about the terminal state?
            state, reward = self.problem.step((swap_i, swap_j))

            sum_r += reward

            if t == 0:
                R = reward.unsqueeze(0)
            else:
                R = torch.cat([R, reward.unsqueeze(0)], dim=0)

            ep.write(action=(swap_i, swap_j), reward=R[-1])
            t += 1

        ep.wrap()

        self.experience_replay_buffer.append(ep)
        self.log.add_item('tot_return', sum_r.item())
        tot_return = R.sum().item()

        return R, tot_return

    def sample_from_buffer(self, batch_size, q_step, gnn_step, episode_len, ddqn):

        # sample #batch_size indices in replay buffer
        idx = np.random.choice(range(len(self.experience_replay_buffer) * episode_len), size=batch_size, replace=True)
        # locate samples in each episode
        batch_idx = [(i // episode_len, i % episode_len) for i in idx]
        # locate start/end states for each sample
        idx_start = [i for i in batch_idx if i[1] % episode_len < episode_len - q_step]

        t = 0
        R = []
        Q = []
        for episode_i, step_j in idx_start:
            # TODO: should run in parallel and avoid the for-loop (need to forward graph batches in dgl)

            # calculate start/end states
            if self.cuda:
                G_start = to_cuda(self.experience_replay_buffer[episode_i].init_state)
                G_end = to_cuda(self.experience_replay_buffer[episode_i].init_state)
            else:
                G_start = dc(to_cuda(self.experience_replay_buffer[episode_i].init_state))
                G_end = dc(to_cuda(self.experience_replay_buffer[episode_i].init_state))

            G_start.ndata['label'] = G_start.ndata['label'][self.experience_replay_buffer[episode_i].label_perm[step_j], :]
            G_end.ndata['label'] = G_end.ndata['label'][self.experience_replay_buffer[episode_i].label_perm[step_j+q_step], :]

            # estimate Q-values
            _, _, _, Q_s1a = self.model(G_start, gnn_step=gnn_step, max_step=episode_len, remain_step=episode_len-1-step_j)
            _, _, _, Q_s2a = self.model_target(G_end, gnn_step=gnn_step, max_step=episode_len, remain_step=episode_len-1-step_j-q_step)

            # calculate accumulated reword
            swap_i, swap_j = self.experience_replay_buffer[episode_i].action_seq[step_j]

            r = self.experience_replay_buffer[episode_i].reward_seq[step_j: step_j + q_step]
            r = torch.sum(r * torch.tensor([self.gamma ** i for i in range(q_step)]))

            # calculate diff between Q-values at start/end
            if step_j == episode_len - 1:
                q = 0
            else:
                if not ddqn:
                    q = self.gamma ** q_step * Q_s2a.max()
                else:
                    q = self.gamma ** q_step * Q_s2a[Q_s1a.argmax()]
            q -= Q_s1a[swap_i * G_start.number_of_nodes() + swap_j]

            R.append(r.unsqueeze(0))
            Q.append(q.unsqueeze(0))

            t += 1
        return torch.cat(R), torch.cat(Q)


    def back_loss(self, R, Q, update_model=True):
        print('actual batch size:', R.shape.numel())
        if self.cuda:
            R = R.cuda()
        L = torch.pow(R + Q, 2).sum()
        L.backward(retain_graph=True)
        self.Q_err += L.item()
        if update_model:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.log.add_item('Q_error', self.Q_err)
            self.Q_err = 0
            self.log.add_item('entropy', 0)

    def train_dqn(self, batch_size=16, grad_accum=10, num_episodes=10, episode_len=50, gnn_step=10, q_step=1, ddqn=False):
        """
        :param batch_size:
        :param num_episodes:
        :param episode_len: #steps in each episode
        :param gnn_step: #iters when running gnn
        :param q_step: reward delay step
        :param ddqn: train in ddqn mode
        :return:
        """
        mean_return = 0
        for i in range(num_episodes):
            [_, tot_return] = self.run_episode(gnn_step=gnn_step, episode_len=episode_len)
            mean_return = mean_return + tot_return
        # trim experience replay buffer
        self.trim_replay_buffer()

        for i in range(grad_accum):
            R, Q = self.sample_from_buffer(batch_size=batch_size, q_step=q_step, gnn_step=gnn_step, episode_len=episode_len, ddqn=ddqn)
            self.back_loss(R, Q, update_model=(i % grad_accum == grad_accum - 1))
            del R, Q
            torch.cuda.empty_cache()

        return self.log

    def trim_replay_buffer(self):
        if len(self.experience_replay_buffer) > self.replay_buffer_max_size:
            self.experience_replay_buffer = self.experience_replay_buffer[-self.replay_buffer_max_size:]

    def update_target_net(self):
        self.model_target = pickle.loads(pickle.dumps(self.model))


