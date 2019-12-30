from DQN import DQN, to_cuda, EpisodeHistory
from test import *
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch
import os
import gc
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
problem = KCut_DGL(k=5, m=6, adjacent_reserve=20, hidden_dim=16, random_init_label=True, a=1)
alg = DQN(problem, gamma=0.9, eps=0.1, lr=.02, replay_buffer_max_size=10, cuda_flag=True)

def run_dqn():
    for i in tqdm(range(100)):
        T1 = time.time()
        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log = alg.train_dqn(batch_size=128, grad_accum=20, num_episodes=1, episode_len=50, gcn_step=20, q_step=1, ddqn=False)
        if i % 3 == 0:
            alg.update_target_net()
        T2 = time.time()
        # print('Epoch: {}. T: {}'.format(i, np.round(T2-T1,3)))
        print('Epoch: {}. R: {}. Q error: {}. H: {}. T: {}'.format(i, np.round(log.get_current('tot_return'),2),np.round(log.get_current('Q_error'),3),np.round(log.get_current('entropy'),3),np.round(T2-T1,3)))

run_dqn()

def test_model(step=100):
    test_problem = KCut_DGL(k=5, m=6, adjacent_reserve=20, hidden_dim=16, random_init_label=True, a=1)
    n = test_problem.k * test_problem.m
    g = to_cuda(test_problem.g)
    ep = EpisodeHistory(g, max_episode_len=step)

    for i in range(step):
        S_a_encoding, h, Q_sa = alg.model(g, step=20)
        best_action = Q_sa.argmax()
        swap_i, swap_j = best_action / n, best_action - best_action / n * n
        state, reward = test_problem.step((swap_i, swap_j))
        ep.write((swap_i, swap_j), reward)
        g = to_cuda(state)

    return ep

ep = test_model(100)

sum(ep.reward_seq)

