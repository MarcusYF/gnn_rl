# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:30:53 2019

@author: fy4bc
"""

from DQN import DQN
from k_cut import *
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch
import os
import gc
import memory_profiler
import matplotlib.pyplot as plt
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

def vis_g(problem, name='test', topo='knn'):
    k = problem.k
    g = problem.g
    X = g.ndata['x']
    n = X.shape[0]
    label = g.ndata['label']
    link = dc(g.edata['e_type'].view(n, n - 1, 2))
    c = ['r', 'b', 'y']
    plt.cla()

    for i in range(k):
        a = X[(label[:, i] > 0).nonzero().squeeze()]
        plt.scatter(a[:, 0], a[:, 1], s=60, c=c[i])

    for i in range(n):
        plt.annotate(str(i), xy=(X[i, 0], X[i, 1]))
        for j in range(n - 1):
            if topo=='knn':
                topo = 0
            else:
                topo = 1
            if link[i, j][topo].item() == 1:
                j_ = j
                if j >= i:
                    j_ = j + 1
                plt.plot([X[i, 0], X[j_, 0]], [X[i, 1], X[j_, 1]], '-', color='k')

    plt.savefig('./' + name + '.png')
    plt.close()


from k_cut import KCut_DGL
k=3
problem = KCut_DGL(k=k, m=3, adjacent_reserve=5, hidden_dim=8, random_init_label=True, a=1, sample=False)
problem = KCut_DGL(k=k, m=3, adjacent_reserve=5, hidden_dim=8, random_init_label=False, a=1, sample=True
                   , label=[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
                   , x=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [0.7, 0.8], [1.5, 1.5], [1.5, 3], [3, 3], [3, 1.5], [1.8, 1.7], [1.8, 0.3], [1.8, 0.8], [2.3, 0.8], [2.3, 0.3], [2.1, 0.5]]))
problem.calc_S()
problem.g.ndata['label']
vis_g(problem, name='toy9', topo='cluster')
problem.reset_label([0,1,1,0,1,2,2,0,2])
problem.calc_S()
problem.step((0,1))
# g = generate_G(k=k, m=4, adjacent_reserve=1, hidden_dim=8, random_init_label=True, a=1, sample=False)['g']

import itertools
from tqdm import tqdm
import numpy as np
all_state = set([','.join([str(i) for i in x]) for x in list(itertools.permutations([0,0,0,1,1,1,2,2,2], 9))])

Q_table = {}
c = 0
for state in all_state:
    c += 1
    print(c)
    l = [int(x) for x in state.split(',')]
    R = []
    for i in range(9):
        for j in range(9):
            problem.reset_label(l)
            _, r = problem.step((i, j))
            R.append(r.item())
    Q_table[state] = np.array(R)


# Q-value iter
# problem_bak = dc(problem) # detached backup for problem instance
# Q_table_bak = dc(Q_table) # detached backup for Q-table


Q_table_ = dc(Q_table_bak) # target Q-table
Q_table = dc(Q_table_bak) # Q-table
State_Action_Reward = dc(Q_table_bak)
gamma = 1.0


Err = []
for ite in tqdm(range(10)):
    for state in Q_table.keys():
        for i in range(9):
            for j in range(9):
                action = 9*i+j
                s = state.split(',')
                swap_i = s[i]
                swap_j = s[j]
                s[i] = swap_j
                s[j] = swap_i
                new_state = ','.join(s)
                Q_table[state][action] = State_Action_Reward[state][action] + gamma * max(Q_table_[new_state])

    # update target Q-table
    err = 0
    for state in Q_table.keys():
        diff = Q_table[state] - Q_table_[state]
        err += np.sqrt(np.mean(diff ** 2))
    Err.append(err)
    print(err)
    Q_table_ = dc(Q_table)



problem.reset_label([2,1,1,2,0,1,0,0,2])
problem.calc_S()
vis_g(problem, name='test9', topo='cluster') 

action = np.argmax(Q_table['0,1,1,2,0,2,0,1,2'])
action = (action // 9, action % 9)
_, reward1 = problem.step(action)
vis_g(problem, name='test9_1', topo='cluster')

action = np.argmax(Q_table['0,2,1,2,0,1,0,1,2'])
action = (action // 9, action % 9)
_, reward2 = problem.step(action)
vis_g(problem, name='test9_2', topo='cluster')

action = np.argmax(Q_table['0,0,1,2,0,1,2,1,2'])
action = (action // 9, action % 9)
_, reward3 = problem.step(action)
vis_g(problem, name='test9_3', topo='cluster')

action = np.argmax(Q_table['0,0,1,2,1,1,2,0,2'])
action = (action // 9, action % 9)
_, reward3 = problem.step(action)
vis_g(problem, name='test9_3', topo='cluster')


for state in Q_table.keys():
    if max(Q_table[state]) <= 0.02:
        print(state)

g = problem.g
vis_g(problem, name='test', topo='cluster')
problem.step((4,8))
vis_g(problem, name='test_1', topo='cluster')
problem.step((7,11))
vis_g(problem, name='test_2', topo='cluster')
problem.step((5,6))
vis_g(problem, name='test_3', topo='cluster')
problem.step((2,10))
vis_g(problem, name='test_4', topo='cluster')

# Q-value iteration


g = dc(problem.g)
problem.step((4,10), g)
g = dc(problem.g)
problem.step((4,10), g)
