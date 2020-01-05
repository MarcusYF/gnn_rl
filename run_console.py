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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
k = 3
m = 3
ajr = 5
hidden_dim = 16
a = 1
gamma = 0.90
lr = 1e-3
replay_buffer_max_size = 100
n_epoch = 100
eps = np.linspace(1.0, 0.05, n_epoch/2)
target_update_step = 5
batch_size = 100
grad_accum = 10
num_episodes = 20
episode_len = 50
gnn_step = 3
q_step = 1
ddqn = False

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
alg = DQN(problem, gamma=gamma, eps=0.1, lr=lr, replay_buffer_max_size=replay_buffer_max_size, cuda_flag=True)


def run_dqn():
    for i in tqdm(range(n_epoch)):

        if i > len(eps) - 1:
            alg.eps = eps[-1]
        else:
            alg.eps = eps[i]

        T1 = time.time()
        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log = alg.train_dqn(batch_size=batch_size, grad_accum=grad_accum, num_episodes=num_episodes, episode_len=episode_len, gnn_step=gnn_step, q_step=q_step, ddqn=ddqn)
        if i % target_update_step == target_update_step - 1:
            alg.update_target_net()
        T2 = time.time()
        # print('Epoch: {}. T: {}'.format(i, np.round(T2-T1,3)))
        print('Epoch: {}. R: {}. Q error: {}. H: {}. T: {}'.format(i, np.round(log.get_current('tot_return'),2),np.round(log.get_current('Q_error'),3),np.round(log.get_current('entropy'),3),np.round(T2-T1,3)))

run_dqn()

def test_model(episode_len=50, explore_prob=0.1):
    test_problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
    S = test_problem.calc_S()
    n = test_problem.k * test_problem.m
    g = to_cuda(test_problem.g)
    ep = EpisodeHistory(g, max_episode_len=episode_len)
    for i in range(episode_len):
        S_a_encoding, h1, h2, Q_sa = alg.model(g, gnn_step=gnn_step, max_step=episode_len, remain_step=episode_len-1-i)
        # epsilon greedy strategy
        if torch.rand(1) > explore_prob:
            best_action = Q_sa.argmax()
        else:
            best_action = torch.randint(high=9, size=(1,)).squeeze()
        swap_i, swap_j = best_action / n, best_action - best_action / n * n
        state, reward = test_problem.step((swap_i, swap_j))
        ep.write((swap_i, swap_j), reward)
        g = to_cuda(state)

    return S, ep

res_S = []
res_gain = []
res_ratio = []
for i in tqdm(range(100)):
    S, ep = test_model(episode_len=episode_len, explore_prob=.1)
    res_S.append(S)
    res_gain.append(sum(ep.reward_seq))
    res_ratio.append(res_gain[-1]/S)

sum(res_gain)
sum(res_ratio)