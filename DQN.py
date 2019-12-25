# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:39 2018

@author: orrivlin
"""
import torch
import numpy as np
import dgl
import torch.nn.functional as F
from copy import deepcopy as dc
import os
from test import *
from Utils import stack_indices, stack_list_indices, max_graph_array
from log_utils import mean_val, logger
from memory_profiler import profile
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

g = generate_G(k=3, m=5, adjacent_reserve=7, hidden_dim=6, a=1, sample=False)
model = DQNet(k=3, m=5, ajr=7, num_head=4, hidden_dim=6)
# iter GCN for fixed steps and forward dqn
S_a_encoding, h, Q_sa = model(g['g'], step=10)

@profile
class DQN:
    def __init__(self, problem, gamma=1.0, eps=0.1, lr=1e-4, cuda_flag=True):
        self.problem = problem
        self.G = problem.g
        self.k = problem.k
        self.m = problem.m
        self.ajr = problem.adjacent_reserve
        self.hidden_dim = problem.hidden_dim
        self.n = self.k * self.m
        self.eps = eps
        if cuda_flag:
            self.model = DQNet(k=self.k, m=self.m, ajr=self.ajr, num_head=4, hidden_dim=self.hidden_dim).cuda()
        else:
            self.model = DQNet(k=self.k, m=self.m, ajr=self.ajr, num_head=4, hidden_dim=self.hidden_dim)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.experience_replay_buffer = []
        self.replay_buffer_max_size = 300
        self.cuda = cuda_flag
        self.log = logger()
        self.log.add_log('tot_return')
        self.log.add_log('TD_error')
        self.log.add_log('entropy')

    def run_episode(self, gcn_step=10, episode_len=50):
        sum_r = 0
        state = self.problem.reset()
        t = 0

        temp_buffer = []

        while t < episode_len:
            G = dc(state) # TODO might cause memory problem
            if self.cuda:
                G.ndata['x'] = G.ndata['x'].cuda()
                G.ndata['label'] = G.ndata['label'].cuda()
                G.ndata['h'] = G.ndata['h'].cuda()
                G.edata['d'] = G.edata['d'].cuda()
                G.edata['w'] = G.edata['w'].cuda()
                G.edata['e_type'] = G.edata['e_type'].cuda()

            S_a_encoding, h, Q_sa = self.model(G, step=gcn_step)

            # epsilon greedy strategy
            if torch.rand(1) > self.eps:
                best_action = Q_sa.argmax()
            else:
                best_action = torch.randint(high=self.n*self.n, size=(1,)).squeeze()
            swap_i, swap_j = best_action/self.n, best_action-best_action/self.n*self.n
            state, reward = self.problem.step((swap_i, swap_j))

            sum_r += reward
            q = Q_sa[best_action].unsqueeze(0)

            if t==0:
                A = [(swap_i, swap_j)]
                R = reward.unsqueeze(0)
                Q = q
            else:
                A.append((swap_i, swap_j))
                R = torch.cat([R, reward.unsqueeze(0)], dim=0)
                Q = torch.cat([Q, q], dim=0)

            # TODO should we also store G
            # self.experience_replay_buffer.append((G, (swap_i, swap_j), reward, q, dc(state)))
            temp_buffer.append([t, reward, q.squeeze(0)])
            # del G
            # gc.collect()
            t += 1

        '''
            [(r0, q0), (r1,   q1), (r2,          q2), ...] ->
            [ (0, q0), (r0, r*q1), (r0+r*r1, r^2*q2), ...]
        '''
        for i in range(1, len(temp_buffer)):
            temp_buffer[i][1] = temp_buffer[i-1][1] + temp_buffer[i][1] * self.gamma ** i
            temp_buffer[i][2] *= self.gamma ** i
        for i in range(1, len(temp_buffer)):
            temp_buffer[-i][1] = temp_buffer[-i-1][1]

        self.experience_replay_buffer.extend(temp_buffer)


        self.log.add_item('tot_return', sum_r.item())
        tot_return = R.sum().item()
        # for i in range(R.shape[0] - 1):
        #     R[-2-i] = R[-2-i] + self.gamma*R[-1-i]

        return R, Q, A, tot_return

    def update_model(self, R, Q):
        self.optimizer.zero_grad()
        if self.cuda:
            R = R.cuda()
        L = 0
        for i in range(R.shape[-1]-1):
            L += torch.pow(Q[:, i] - R[:, i] - self.gamma * Q[:, i+1], 2).sum()
        L.backward(retain_graph=True)
        self.optimizer.step()
        self.log.add_item('TD_error', L.detach().item())
        self.log.add_item('entropy', 0)

    def update_model_dqn(self, R, Q):
        print('actual batch size:', R.shape.numel())
        self.optimizer.zero_grad()
        if self.cuda:
            R = R.cuda()
        L = torch.pow(R + Q, 2).sum()
        L.backward(retain_graph=True)
        self.optimizer.step()
        self.log.add_item('TD_error', L.detach().item())
        self.log.add_item('entropy', 0)


    def train(self, num_episodes=10, episode_len=50, gcn_step=10):
        mean_return = 0
        for i in range(num_episodes):
            [r, q, _, tot_return] = self.run_episode(gcn_step=gcn_step, episode_len=episode_len)
            mean_return = mean_return + tot_return
            if i == 0:
                R = r.unsqueeze(0)
                Q = q.unsqueeze(0)
            else:
                R = torch.cat([R, r.unsqueeze(0)], dim=0)
                Q = torch.cat([Q, q.unsqueeze(0)], dim=0)

        mean_return = mean_return / num_episodes
        self.update_model(R, Q)
        return self.log

    def train_dqn(self, batch_size=16, num_episodes=10, episode_len=50, gcn_step=10, q_step=1):
        mean_return = 0
        for i in range(num_episodes):
            [_, _, _, tot_return] = self.run_episode(gcn_step=gcn_step, episode_len=episode_len)
            mean_return = mean_return + tot_return
        # trim experience replay buffer
        self.trim_replay_buffer()
        # sample batch from experience replay buffer
        ind = np.random.choice(range(len(self.experience_replay_buffer)), size=batch_size, replace=True)
        ind_start = [i for i in ind if i % episode_len < episode_len - q_step]

        t = self.experience_replay_buffer[ind_start[0]][0]
        R = (self.experience_replay_buffer[ind_start[0]][1].unsqueeze(0)
        - self.experience_replay_buffer[ind_start[0]+q_step][1].unsqueeze(0)) / (self.gamma ** t)
        Q = self.experience_replay_buffer[ind_start[0]][2].unsqueeze(0)
        - self.experience_replay_buffer[ind_start[0]+q_step][2].unsqueeze(0) / (self.gamma ** t)
        for i in ind_start[1:]:
            t = self.experience_replay_buffer[i][0]
            r = (self.experience_replay_buffer[i][1] - self.experience_replay_buffer[i+q_step][1]) / (self.gamma ** t)
            q = (self.experience_replay_buffer[i][2] - self.experience_replay_buffer[i+q_step][2]) / (self.gamma ** t)
            R = torch.cat([R, r.unsqueeze(0)], dim=0)
            Q = torch.cat([Q, q.unsqueeze(0)], dim=0)

        mean_return = mean_return / num_episodes
        self.update_model_dqn(R, Q)
        return self.log

    def trim_replay_buffer(self):
        if len(self.experience_replay_buffer) > self.replay_buffer_max_size:
            self.experience_replay_buffer = self.experience_replay_buffer[-self.replay_buffer_max_size:]

