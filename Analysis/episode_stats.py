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
from toy_models.Qiter import vis_g

problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)

def test_model(problem, episode_len=50, explore_prob=0.1):
    problem.reset()
    test_problem = problem
    S = test_problem.calc_S()
    g = to_cuda(test_problem.g)
    ep = EpisodeHistory(g, max_episode_len=episode_len)
    for i in range(episode_len):
        legal_actions = test_problem.get_legal_actions()
        S_a_encoding, h1, h2, Q_sa = alg.model(g, legal_actions, gnn_step=gnn_step)
        # epsilon greedy strategy
        if torch.rand(1) > explore_prob:
            action_idx = Q_sa.argmax()
        else:
            action_idx = torch.randint(high=legal_actions.shape[0], size=(1,)).squeeze()
        swap_i, swap_j = legal_actions[action_idx]
        state, reward = test_problem.step((swap_i, swap_j))
        ep.write((swap_i, swap_j), action_idx, reward)
        g = to_cuda(state)

    return S, ep


class test_summary():

    def __init__(self, problem=problem, num_instance=100):
        self.problem = problem
        self.num_instance = num_instance
        self.S = []
        self.gain = []
        self.gain_ratio = []

    def run_test(self, episode_len=10, explore_prob=.0):

        for i in tqdm(range(self.num_instance)):
            self.problem.reset()
            s, ep = test_model(self.problem, episode_len=episode_len, explore_prob=explore_prob)
            self.S.append(s)
            self.gain.append(sum(ep.reward_seq))
            self.gain_ratio.append(self.gain[-1] / s)

    def show_result(self):

        print('Avg value of initial S:', np.mean(self.S))
        print('Avg gain:', np.mean(self.gain))
        print('Avg percentage gain:', np.mean(self.gain_ratio))
        print('Percentage of instances with positive gain:', len([x for x in self.gain if x > 0]) / self.num_instance)


problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
baseline = test_summary(problem=problem, num_instance=1000)
baseline.run_test(explore_prob=1.0)
baseline.show_result()

test1 = test_summary(problem=problem, num_instance=1000)
test1.run_test()
test1.show_result()

buf = alg.experience_replay_buffer[0]

buf.action_seq
buf.action_indices = [tensor(14), tensor(15), tensor(0), tensor(1), tensor(13)]
buf.reward_seq = tensor([ 0.4035, -0.4050,  0.5623, -0.2317,  0.3374])
