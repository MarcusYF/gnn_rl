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

    # need to adapt from Canonical_solvers.py
    def cmpt_optimal(self):
        return

    # need to adapt from Canonical_solvers.py
    def test_greedy(self):
        return

    def test_model(self, gnn_step=3, episode_len=50, explore_prob=0.1, time_aware=False):
        self.problem.reset()
        test_problem = self.problem
        S = test_problem.calc_S()
        g = to_cuda(test_problem.g)
        ep = EpisodeHistory(g, max_episode_len=episode_len)
        for i in range(episode_len):
            legal_actions = test_problem.get_legal_actions()

            S_a_encoding, h1, h2, Q_sa = self.alg.model(dgl.batch([g]), legal_actions.cuda(), gnn_step=gnn_step,
                                                   time_aware=time_aware, remain_episode_len=episode_len - i - 1)

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

    def run_test(self, episode_len=50, explore_prob=.0, time_aware=False, criteria='end'):

        for _ in tqdm(range(self.num_instance)):
            self.problem.reset()
            s, ep = self.test_model(episode_len=episode_len, explore_prob=explore_prob, time_aware=time_aware)
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






# buf = alg.experience_replay_buffer[-1]
#
# g1 = problem.g
# g1a = problem.get_legal_actions()
# problem.reset()
# g2 = problem.g
# g2a = problem.get_legal_actions()
# bg = dgl.batch([g1, g2])
# model = model.cpu()
# sa1, _, h1, q1 = model(g1, g1a)
# sa2, _, h2, q2 = model(g2, g2a)
#
# ga = torch.cat([g1a, g2a], axis=0)
#
# torch.norm(torch.cat([sa1, sa2], axis=0) - sa)
# torch.norm(torch.cat([h1, h2], axis=0) - h)
# torch.norm(torch.cat([q1, q2], axis=0) - q)
# sa, _, h, q = model(bg, ga)
#
# a = conv(g1, aa)
# b = conv(g2, bb)
# c = conv(bg, torch.cat([aa, bb], axis=0))

# baseline = test_summary(alg=alg, problem=problem, num_instance=100)
# baseline.run_test(explore_prob=1.0)
# baseline.show_result()
folder = 'Models/dqn_0114_base/'
# folder = 'Models/dqn_test_not_sample_batch_episode/'
with open(folder + 'dqn_' + str(5000), 'rb') as model_file:
    alg = pickle.load(model_file)
problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
test1 = test_summary(alg=alg, problem=problem, num_instance=100)
test1.run_test(episode_len=50, explore_prob=0.0, time_aware=False)
test1.show_result()

test2 = test_summary(alg=alg, problem=problem, num_instance=100)
test2.run_test(episode_len=50, explore_prob=0, time_aware=False)
test2.show_result()


problem.calc_S(alg.experience_replay_buffer[1].init_state)
y = 0
z = 0
for i in range(test1.S.__len__()):
    s = problem.calc_S(alg.experience_replay_buffer[i].init_state)
    traj = 1 - np.cumsum(alg.experience_replay_buffer[i].reward_seq) / s
    y += torch.cat([torch.tensor([1.]), traj], axis=0)
for i in range(test2.S.__len__()):
    s = test2.S[i]
    traj = 1 - np.cumsum(test2.episodes[i].reward_seq) / s
    a = [1]
    a.extend(list(traj))
    z += np.array(a)
plt.figure(figsize=[10, 5])
plt.subplot(121)
plt.plot(y/100)
plt.xlabel('action steps')
plt.ylabel('avg. gain in S(%)')
plt.title('training')
plt.subplot(122)
plt.plot(z/100)
plt.xlabel('action steps')
plt.ylabel('avg. gain in S(%)')
plt.title('testing')
plt.savefig('Analysis/' + 'trajectory_vis_2' + '.png')
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




x = []
for i in tqdm(range(100)):
    buf = alg_q_110.experience_replay_buffer[i]
    x.append(sum(buf.reward_seq))
np.argmax(x)

buf = alg_q_110.experience_replay_buffer[39]
buf.action_seq
buf.reward_seq

alg.run_episode(gnn_step=3)

buf = alg.experience_replay_buffer[0]
buf.action_seq
buf.reward_seq

[(tensor(6, device='cuda:0'), tensor(7, device='cuda:0')),
 (tensor(0, device='cuda:0'), tensor(6, device='cuda:0')),
 (tensor(4, device='cuda:0'), tensor(5, device='cuda:0')),
 (tensor(4, device='cuda:0'), tensor(7, device='cuda:0')),
 (tensor(5, device='cuda:0'), tensor(8, device='cuda:0')),
 (tensor(1), tensor(8)),
 (tensor(0), tensor(1)),

problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
buf = alg.experience_replay_buffer[0]
buf.init_state.ndata['label']
problem.g = dc(buf.init_state)

l = buf.init_state.ndata['label'][[8, 4, 3, 2, 6, 7, 1, 5, 0],:].nonzero()[:,1]

problem.reset_label([0, 1, 0, 2, 2, 1, 2, 1, 0])
problem.calc_S()
problem.step((1,8))
vis_g(problem, name='toy_models/0109_4', topo='c')



sa_,_,_,Q_sa_=alg.model(to_cuda(problem.g), problem.get_legal_actions())
[-0.9731,  0.4071, -0.2058,  0.1090,  0.7073, -0.7521,  0.7732, -1.1587,
         0.1465,  1.4894, -0.4074, -1.0282,  0.4630, -0.3669, -0.1107, -0.0029,
         0.0052,  0.4920, -0.1909,  0.0461,  0.9827,  0.3718, -0.4183, -0.2039,
         0.2517, -0.3922,  0.0429,  0.0043, -0.0671, -0.3431,  0.3735, -0.0446,
        -0.9731,  0.0697,  0.4810,  0.1089, -0.0939,  0.1573, -0.2550,  0.4802,
        -0.2886, -0.2930,  0.0931,  0.5797, -0.4734, -0.3426, -0.1073,  0.4440,
        -0.1909, -0.3064]
problem.step((4,8))
_,_,_,Q_sa1=alg.model(to_cuda(problem.g), problem.get_legal_actions())
[ 2.6993e-04,  1.5106e-01,  2.0658e-01,  2.2934e-02,  1.8874e-01,
         1.8800e-01, -7.8708e-03,  1.6216e-01,  2.1767e-01,  1.4793e-02,
         1.9973e-01,  1.9841e-01,  1.6251e-01,  1.2036e-01,  1.2011e-01,
         1.5385e-01,  2.1907e-01,  1.3744e-01,  1.3767e-01,  1.7969e-01,
         1.4820e-01,  *3.0644e-01*,  1.3666e-01,  1.3641e-01,  1.7007e-01,
         2.3731e-01,  2.3643e-01]

[1,2,2,0,0,1,1,0,2]
problem.calc_S() 4.1640
[2, 6], [0, 3], [0, 3], [4, 6], [1, 6]
[ 0.9560, -0.6474,  0.6474, -0.3616,  0.0909, -0.9082,  0.9082, -0.9082,
         0.9082,  0.6987, -1.0706,  0.3291,  0.9592, -1.1792,  0.3503, -0.3503,
         0.3503, -0.2162,  0.1179,  0.0304]
problem.reset_label([1,2,2,0,0,1,1,0,2])
problem.step((0,3))
buf.action_indices
problem.calc_S()

x = []
for i in range(100):
    x.append(alg.experience_replay_buffer[i].reward_seq[0])
