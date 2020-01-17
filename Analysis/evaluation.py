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
# import Analysis.episode_stats.test_summary,test_model

# folder = 'Models/dqn_0113_test_qstep/'
folder = 'Models/dqn_0114_base/'
# folder = 'Models/dqn_test_not_sample_batch_episode/'
folder = 'Models/dqn_0113_cutloop/'
# folder = 'Models/dqn_0113_test_eps0/'
# folder = 'Models/dqn_test_centroid_h16/'
with open(folder + 'dqn_' + str(10000), 'rb') as model_file:
    alg = pickle.load(model_file)

x = []
for i in range(alg.experience_replay_buffer.__len__()):
    x.append(sum(alg.experience_replay_buffer[i].reward_seq))
sum(x)

# scp -r /u/fy4bc/code/research/RL4CombOptm/gnn_rl/Models/dqn_0113_test_eps0 fy4bc@128.143.69.125:/home/fy4bc/mnt/code/research/RL4CombOptm/MinimumVertexCover_DRL/Models/

13.7886
35.3720
64.9265
85.8639
106.3571
97.5171
109.9404
106.3062

# plot Q-loss/Reward curve
fig_name = 'return-base-8'

ret = alg.log.get_log("tot_return")
qv = alg.log.get_log("Q_error")
x = []
for i in range(len(qv)):
    if i*10+100<=len(ret):
        x.append(np.mean(ret[i*10:i*10+100]))

fig = plt.figure(figsize=[15, 5])
ax = fig.add_subplot(121)
ax.plot(x, label='episode reward')
ax2 = ax.twinx()
eps = np.concatenate([np.linspace(1.0, 0.1, 1000), np.ones(len(qv)-1000)*0.1]) #
# eps = np.linspace(0.9, 0.1, 1000)
ax2.plot(eps, label='\epsilon-greedy exploration prob.', color='r')
# fig.legend(loc=1)
ax.set_xlabel('Training Epochs')
ax.set_ylabel("Accumulated Episode Reward")
ax2.set_ylabel("Exploration Probability")
ax.set_title('Training Reward')
ax = fig.add_subplot(122)
ax.plot(range(9999), qv[0:], label='episode reward')
ax2 = ax.twinx()
ax2.plot(eps, label='\epsilon-greedy exploration prob.', color='r')
# fig.legend(loc=1)
ax.set_xlabel('Training Epochs')
ax.set_ylabel("Qradratic Q-loss ")
ax2.set_ylabel("Exploration Probability")
ax.set_title('Training Loss')
plt.savefig('./Analysis/' + fig_name + '.png')
plt.close()

## plot test performance curve
buf_perf = 0 # buffer performance
for i in range(test1.S.__len__()):
    s = problem.calc_S(alg.experience_replay_buffer[i].init_state)
    traj = 1 - np.cumsum(alg.experience_replay_buffer[i].reward_seq) / s
    buf_perf += torch.cat([torch.tensor([1.]), traj], axis=0)
tst_perf = 0 # test performance
for i in range(test1.S.__len__()):
    s = test1.S[i]
    traj = 1 - np.cumsum(test1.episodes[i].reward_seq) / s
    a = [1]
    a.extend(list(traj))
    tst_perf += np.array(a)
grd_perf = 0 # greedy algorithm performance/from toy_models/Canonical_solvers
for i in range(100):
    s = res[0, i]
    traj = 1 - np.cumsum(greedy_move_history[i]) / s
    a = [1]
    a.extend(list(traj))
    a = a + [a[-1]] * (51-len(a))
    grd_perf += np.array(a)

performance_horizon = [1-(sum(res[0,:])-sum(res[1,:]))/sum(res[0,:])] * 51


plt.figure(figsize=[10, 10])
plt.subplot(111)
plt.plot(buf_perf/100, color='y')
plt.plot(tst_perf/100, color='b')
plt.plot(grd_perf/100, color='r')
plt.plot(performance_horizon, color='k')
plt.legend(['replay buffer(eps=0.1)', 'test set(eps=0)', 'greedy search', 'optimal solution'], loc="upper right", fontsize=20)
plt.xlabel('Action Steps', fontsize=20)
plt.ylabel('Scaled Objective-S', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Agent Performance and Benchmarks', fontsize=20)
plt.savefig('Analysis/' + 'trajectory_vis_3' + '.png')
plt.close()

# scp -r fy4bc@128.143.69.125:/home/fy4bc/mnt/code/research/RL4CombOptm/MinimumVertexCover_DRL/Data/qiter33 /u/fy4bc/code/research/RL4CombOptm/gnn_rl/Data
