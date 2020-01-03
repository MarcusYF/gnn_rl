from DQN import DQN, to_cuda, EpisodeHistory
from k_cut import *
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch
import os
import gc
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
k = 3
m = 3
ajr = 5
hidden_dim = 8
a = 1
gamma = 0.95
eps = 0.1
lr = 0.0001
replay_buffer_max_size = 10
n_epoch = 50
target_update_step = 5
batch_size = 100
grad_accum = 10
num_episodes = 1
episode_len = 50
gnn_step = 4
q_step = 1
ddqn = False

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim, random_init_label=True, a=a)
alg = DQN(problem, gamma=gamma, eps=eps, lr=lr, replay_buffer_max_size=replay_buffer_max_size, cuda_flag=True)

def run_dqn():
    for i in tqdm(range(n_epoch)):
        T1 = time.time()
        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log = alg.train_dqn(batch_size=batch_size, grad_accum=grad_accum, num_episodes=num_episodes, episode_len=episode_len, gnn_step=gnn_step, q_step=q_step, ddqn=ddqn)
        if i % target_update_step == target_update_step - 1:
            alg.update_target_net()
        T2 = time.time()
        # print('Epoch: {}. T: {}'.format(i, np.round(T2-T1,3)))
        print('Epoch: {}. R: {}. Q error: {}. H: {}. T: {}'.format(i, np.round(log.get_current('tot_return'),2),np.round(log.get_current('Q_error'),3),np.round(log.get_current('entropy'),3),np.round(T2-T1,3)))

run_dqn()

def test_model(episode_len=50):
    test_problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim, random_init_label=True, a=a)
    n = test_problem.k * test_problem.m
    g = to_cuda(test_problem.g)
    ep = EpisodeHistory(g, max_episode_len=episode_len)

    for i in range(episode_len):
        S_a_encoding, h1, h2, Q_sa = alg.model(g, gnn_step=gnn_step, max_step=episode_len, remain_step=episode_len-1-i)
        best_action = Q_sa.argmax()
        swap_i, swap_j = best_action / n, best_action - best_action / n * n
        state, reward = test_problem.step((swap_i, swap_j))
        ep.write((swap_i, swap_j), reward)
        g = to_cuda(state)

    return ep

ep = test_model(episode_len)

sum(ep.reward_seq)


g1 = to_cuda(problem.g)
g2 = to_cuda(problem.g)
i=49
S_a_encoding, h1, h2, Q_sa = alg.model(g2, gnn_step=gnn_step, max_step=episode_len, remain_step=1)
Q_sa.argmax()

Q_sa

i=0
S_a_encoding, h1, h2, Q_sa = alg.model(g1, gnn_step=gnn_step, max_step=episode_len)
Q_sa.argmax()
