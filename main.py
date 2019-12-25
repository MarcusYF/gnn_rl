# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:30:53 2019

@author: fy4bc
"""

from DQN import DQN
from test import *
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch
import os
import gc
import memory_profiler
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# n = 30  # number of nodes
# p = 0.15  # edge probability
# env = MVC(n, p)
# alg = DiscreteActorCritic(env, cuda_flag=True)

# n = 20
# grp_size = 5
# env = KCut(n, grp_size)
# alg = DQN(env, cuda_flag=Truea)
# alg = A3C(env, cuda_flag=True)


problem = KCut_DGL(k=5, m=6, adjacent_reserve=20, hidden_dim=16, random_init_label=True, a=1)

problem.S
problem.g.ndata['label']

problem.step((1, 10))
problem.step((2, 1))
problem.step((7, 17))
problem.step((4, 24))
problem.step((19, 4))
# problem.g.ndata['label']
problem.S

k1 = []
k2 = []
k3=[]
for i in range(20):
    a = problem.calc_S(alg.experience_replay_buffer[i][0])
    b = problem.calc_S(alg.experience_replay_buffer[i][3])
    k1.append(a)
    k1.append(b)
    k3.append(alg.experience_replay_buffer[i][2])
    k3.append(b-a)


alg = DQN(problem, gamma=0.9, eps=0.1, lr=.02, cuda_flag=True)

a, b = problem.step((29, 4), g1)
g1 = alg.experience_replay_buffer[0][0]
g2 = alg.experience_replay_buffer[0][3]
g3 = alg.experience_replay_buffer[2][0]
g4 = alg.experience_replay_buffer[2][3]
# model1 = alg.model.cpu()
#
# S_a_encoding, h, Q_sa = model1(problem.g, 1000)
# best_action = Q_sa.argmax()
# swap_i, swap_j = best_action / 30, best_action - best_action / 30 * 30
# swap_i, swap_j
# state, reward = problem.step((swap_i, swap_j))
# reward

@profile
for i in range(50):
    T1 = time.time()
    log = alg.train_ddqn(batch_size=16, num_episodes=5, episode_len=10, gcn_step=10, q_step=1)
    T2 = time.time()
    print('Epoch: {}. R: {}. TD error: {}. H: {}. T: {}'.format(i, np.round(log.get_current('tot_return'),2),np.round(log.get_current('TD_error'),3),np.round(log.get_current('entropy'),3),np.round(T2-T1,3)))

Y = np.asarray(log.get_log('tot_return'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('episode return')

Y = np.asarray(log.get_log('TD_error'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y , Y2)
plt.xlabel('episodes')
plt.ylabel('mean TD error')

Y = np.asarray(log.get_log('entropy'))
Y2 = smooth(Y)
x = np.linspace(0, len(Y), len(Y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, Y, Y2)
plt.xlabel('episodes')
plt.ylabel('mean entropy')

PATH = 'mvc_net.pt'
torch.save(alg.model.state_dict(), PATH)

