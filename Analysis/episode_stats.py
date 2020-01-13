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

        for _ in tqdm(range(self.num_instance)):
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

folder = 'Models/dqn_0112_base_parallel/'
# folder = 'Models/dqn_0113_base_parallel_test_grad_accum/'
with open(folder + 'dqn_' + str(1500), 'rb') as model_file:
    alg = pickle.load(model_file)

x = []
for i in range(alg.experience_replay_buffer.__len__()):
    x.append(sum(alg.experience_replay_buffer[i].reward_seq))
sum(x)

13.7886
35.3720
64.9265
85.8639
106.3571
97.5171
109.9404
106.3062


buf = alg.experience_replay_buffer[-1]

problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
g1 = problem.g
g1a = problem.get_legal_actions()
problem.reset()
g2 = problem.g
g2a = problem.get_legal_actions()
bg = dgl.batch([g1, g2])
model = model.cpu()
sa1, _, h1, q1 = model(g1, g1a)
sa2, _, h2, q2 = model(g2, g2a)

ga = torch.cat([g1a, g2a], axis=0)

torch.norm(torch.cat([sa1, sa2], axis=0) - sa)
torch.norm(torch.cat([h1, h2], axis=0) - h)
torch.norm(torch.cat([q1, q2], axis=0) - q)
sa, _, h, q = model(bg, ga)

a = conv(g1, aa)
b = conv(g2, bb)
c = conv(bg, torch.cat([aa, bb], axis=0))

# baseline = test_summary(alg=alg, problem=problem, num_instance=100)
# baseline.run_test(explore_prob=1.0)
# baseline.show_result()

test1 = test_summary(alg=alg, problem=problem, num_instance=100)
test1.run_test(episode_len=50, explore_prob=0.0, time_aware=False)
test1.show_result()

test2 = test_summary(alg=alg, problem=problem, num_instance=100)
test2.run_test(episode_len=50, explore_prob=0, time_aware=False)
test2.show_result()

for i in range(100):
    if not test2.episodes[i].action_seq[-1] == test2.episodes[i].action_seq[-2]:
        print(i)
# 1 3
# 5 3
# 26 4
# 32 3
# 40 3
# 51 4
# 61 3
# 64 4
# 68 3
# 70 3
# 92 3
# 95 3

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

ret = alg.log.get_log("tot_return")
for i in range(2489):
    x.append(np.mean(alg.log.get_log("tot_return")[i*10:i*10+100]))
qv=alg.log.get_log("Q_error")

fig = plt.figure(figsize=[10, 5])
ax = fig.add_subplot(121)
ax.plot(x, label='episode reward')
ax2 = ax.twinx()
eps = np.concatenate([np.linspace(1.0, 0.1, 1000),np.ones(len(x)-1000)*0.1 ]) #
ax2.plot(eps, label='\epsilon-greedy exploration prob.', color='r')
# fig.legend(loc=1)
ax.set_xlabel('Training Epochs')
ax.set_ylabel("Accumulated Episode Reward")
ax2.set_ylabel("Exploration Probability")
ax.set_title('Training Reward')
ax = fig.add_subplot(122)
ax.plot(qv, label='episode reward')
ax2 = ax.twinx()
ax2.plot(eps, label='\epsilon-greedy exploration prob.', color='r')
# fig.legend(loc=1)
ax.set_xlabel('Training Epochs')
ax.set_ylabel("Qradratic Q-loss ")
ax2.set_ylabel("Exploration Probability")
ax.set_title('Training Loss')
plt.savefig('./Analysis/' + 'return-base-3' + '.png')
plt.close()



x = []
for i in tqdm(range(100)):
    buf = alg_q_110.experience_replay_buffer[i]
    x.append(sum(buf.reward_seq))
np.argmax(x)

buf = alg_q_110.experience_replay_buffer[39]
buf.action_seq
buf.reward_seq

buf = alg.experience_replay_buffer[0]
buf.action_seq
buf.reward_seq

problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
buf = alg.experience_replay_buffer[-1]
buf.init_state.ndata['label']
problem.g = dc(buf.init_state)

l = buf.init_state.ndata['label'][[8, 4, 3, 2, 6, 7, 1, 5, 0],:].nonzero()[:,1]

problem.reset_label([0, 1, 0, 2, 2, 1, 2, 1, 0])
problem.calc_S()
problem.step((0,2))
vis_g(problem, name='toy_models/0109_4', topo='c')



sa_,_,_,Q_sa_=alg.model(to_cuda(problem.g), problem.get_legal_actions())
[-0.0430, -0.2413,  0.0153, -0.0343,  0.5179,  0.3169, -0.5515,  0.1526,
        -0.4617,  0.5946,  0.3297, -0.3749, -0.0220, -0.1082,  0.2090,  0.1211,
         0.0527,  0.3200, -1.2247, -0.1389,  0.2418,  0.3372,  0.2105, -0.1443,
         0.4358, -1.2212,  1.2940, -0.0576,  0.0735, -0.4657,  0.5310, -0.2026,
         0.0486, -0.8602,  0.6603, -0.0188, -0.0522,  0.1204, -0.3271,  0.1766,
        -0.0987,  0.1082, -0.1460,  0.3840,  0.3362, -0.2727, -0.9411,  1.1252,
         0.9971, -0.7746]
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
