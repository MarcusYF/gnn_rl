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
folder = 'Models/dqn_test_not_sample_batch_episode/'
with open(folder + 'dqn_' + str(500), 'rb') as model_file:
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
fig_name = 'return-base-5'

ret = alg.log.get_log("tot_return")
qv = alg.log.get_log("Q_error")
x = []
for i in range(len(qv)):
    if i*10+100<len(ret):
        x.append(np.mean(ret[i*10:i*10+100]))

fig = plt.figure(figsize=[10, 5])
ax = fig.add_subplot(121)
ax.plot(ret, label='episode reward')
ax2 = ax.twinx()
eps = np.concatenate([np.linspace(1.0, 0.1, 3000), np.ones(len(qv)-3000)*0.1 ]) #
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
plt.savefig('./Analysis/' + fig_name + '.png')
plt.close()

