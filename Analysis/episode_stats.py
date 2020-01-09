from DQN import DQN, to_cuda, EpisodeHistory
from k_cut import *
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import pickle
import torch
import os
import gc
from tqdm import tqdm
from toy_models.Qiter import vis_g


def test_model(alg, problem, gnn_step=3, episode_len=50, explore_prob=0.1, time_aware=False):
    problem.reset()
    test_problem = problem
    S = test_problem.calc_S()
    g = to_cuda(test_problem.g)
    ep = EpisodeHistory(g, max_episode_len=episode_len)
    for i in range(episode_len):
        legal_actions = test_problem.get_legal_actions()
        if time_aware:
            S_a_encoding, h1, h2, Q_sa = alg.model(g, legal_actions, gnn_step=gnn_step, remain_episode_len=episode_len-i-1)
        else:
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

    def __init__(self, alg, problem, num_instance=100):
        self.alg = alg
        self.problem = problem
        self.num_instance = num_instance
        self.episodes = []
        self.S = []
        self.max_gain = []
        self.max_gain_budget = []
        self.max_gain_ratio = []

    def run_test(self, episode_len=50, explore_prob=.0, time_aware=False, criteria='end'):

        for i in tqdm(range(self.num_instance)):
            self.problem.reset()
            s, ep = test_model(self.alg, self.problem, episode_len=episode_len, explore_prob=explore_prob, time_aware=time_aware)
            self.S.append(s.item())
            self.episodes.append((ep))
            if criteria == 'max':
                cum_gain = np.cumsum(ep.reward_seq)
                self.max_gain.append(max(cum_gain).item())
                self.max_gain_budget.append(1 + np.argmax(cum_gain).item())
                self.max_gain_ratio.append(self.max_gain[-1] / s)
            else:
                self.max_gain.append(sum(ep.reward_seq).item())
                self.max_gain_budget.append(episode_len)
                self.max_gain_ratio.append(self.max_gain[-1] / s)

    def show_result(self):

        print('Avg value of initial S:', np.mean(self.S))
        print('Avg max gain:', np.mean(self.max_gain))
        print('Avg max gain budget:', np.mean(self.max_gain_budget))
        print('Var max gain budget:', np.std(self.max_gain_budget))
        print('Avg percentage max gain:', np.mean(self.max_gain_ratio))
        print('Percentage of instances with positive gain:', len([x for x in self.max_gain if x > 0]) / self.num_instance)

# save version 2020.1.6 and continue to train alg
# alg_first_work_version = dc(alg)
# alg_q_110 = dc(alg)

with open('Models/dqn_3_3_0/' + 'dqn_' + str(1400), 'rb') as model_file:
    alg = pickle.load(model_file)

problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
# baseline = test_summary(alg=alg, problem=problem, num_instance=100)
# baseline.run_test(explore_prob=1.0)
# baseline.show_result()

test1 = test_summary(alg=alg, problem=problem, num_instance=100)
test1.run_test(episode_len=50, time_aware=False)
test1.show_result()

test1.S
test1.episodes[1].reward_seq
for i in range(test1.S.__len__()):
    traj = test1.S[i] + np.cumsum(test1.episodes[i].reward_seq)
    plt.plot(traj / traj[0])
    plt.savefig('Analysis/' + 'trajectory_vis' + '.png')
plt.close()



x = []
for i in range(100):
    buf = alg.experience_replay_buffer[i]
    best_s = max(np.cumsum(buf.reward_seq))
    x.append(sum(buf.reward_seq))
buf.action_seq
buf.action_indices = [tensor(14), tensor(15), tensor(0), tensor(1), tensor(13)]
buf.reward_seq = tensor([ 0.4035, -0.4050,  0.5623, -0.2317,  0.3374])

alg.experience_replay_buffer[0].action_seq

np.cumsum(alg.experience_replay_buffer[0].reward_seq)

plt.plot(alg.log.get_log('Q_error'))
plt.savefig('./Analysis/' + 'q-loss-1' + '.png')
plt.close()

len(test1.gain)

x = []
for i in tqdm(range(100)):
    buf = alg_q_110.experience_replay_buffer[i]
    x.append(sum(buf.reward_seq))
np.argmax(x)

buf = alg_q_110.experience_replay_buffer[39]
buf.action_seq
buf.reward_seq


