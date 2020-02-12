from k_cut import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm
from Analysis.episode_stats import test_summary

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# folder = 'Models/dqn_0113_test_qstep/'
# folder = 'Models/dqn_test_not_sample_batch_episode/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + '/dqn_0129_adj_weight/'
# folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_0129_test_ajr' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_5by6_0129_base_2' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_10by10_0129_base' + '/'
# folder = 'Models/dqn_0113_test_eps0/'
# folder = 'Models/dqn_test_centroid_h16/'
# folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_0129_all_weight' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_10by10_0204_softdqn' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_3by3_0205' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_3by3_cascade' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_5by6_02031_mlp_bs200' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_3by3_0204_gnn' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_3by3_0204_gnn_fx' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_3by3_0205_softdqn_anneal' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_3by3_0205_softdqn_10' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_5by6_0205_softdqn2' + '/'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + 'dqn_5by6_0207_soft1' + '/'

model_name = 'test2'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/'
folder = folder + model_name + '/'
with open(folder + 'dqn_' + str(10000), 'rb') as model_file:
    alg = pickle.load(model_file)
with open(folder + 'dqn_' + str(10000), 'rb') as model_file:
    alg1 = pickle.load(model_file)
with open(folder + 'dqn_' + str(4000), 'rb') as model_file:
    alg2 = pickle.load(model_file)
with open(folder + 'dqn_' + str(8000), 'rb') as model_file:
    alg3 = pickle.load(model_file)

# PER state distribution check
ckpt = 10000
with open('/p/reinforcement/data/gnn_rl/model/dqn/test3/buffer_' + str(ckpt), 'rb') as model_file:
    bf = pickle.load(model_file)
w = torch.tensor(bf[0]).flatten()
s = torch.tensor(bf[1]).flatten()

_, idx = s.sort(descending=True)
s_ = s[idx]
w_ = w[idx]
path = os.path.abspath(os.path.join(os.getcwd()))
fig_name = 'state-distr-5'
fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
ax.plot(torch.clamp(w_,0,5), label='weight', color='y')
ax2 = ax.twinx()
ax2.plot(s_, label='after 1k epochs', color='r')
ax.set_xlabel('States')
ax2.set_ylabel("K-cut value")
ax.set_ylabel("weight in replay buffer")
ax.set_title('state distribution in replay buffer during training')
plt.legend(loc="lower left")
plt.savefig(path + '/supervised/case_study/state_distr/' + fig_name + '.png')
plt.close()


# replay buffer state distribution check
problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16, sample_episode=10)
init_collection = problem.graph_generator.generate_batch_G(batch_size=5000, hidden_dim=16)
init_collection = dgl.unbatch(init_collection)
s_collection = []
s11_collection = []
s22_collection = []
s33_collection = []
s44_collection = []

a = []
for i in range(50):
    px = 0
    for j in range(100):
        px += problem.calc_S(g=alg1[3][i][j]).item()
    a.append(px)

# for i in range(50):
#     print(i)
#     for j in range(100):
#         s44_collection.append(problem.calc_S(g=alg1[3][i][j]).item())
# del alg1
# torch.cuda.empty_cache()

for i in tqdm(range(5000)):
    s_collection.append(problem.calc_S(g=init_collection[i]).item())
    s11_collection.append(problem.calc_S(g=alg.experience_replay_buffer2[i].s0).item())
    s22_collection.append(problem.calc_S(g=alg1.experience_replay_buffer2[i].s0).item())
    s33_collection.append(problem.calc_S(g=alg2.experience_replay_buffer2[i].s0).item())
    s44_collection.append(problem.calc_S(g=alg3.experience_replay_buffer2[i].s0).item())

s_collection = sorted(s_collection, reverse=True)
s11_collection = sorted(s11_collection, reverse=True)
s22_collection = sorted(s22_collection, reverse=True)
s33_collection = sorted(s33_collection, reverse=True)
s44_collection = sorted(s44_collection, reverse=True)

path = os.path.abspath(os.path.join(os.getcwd()))
fig_name = 'state-distr-2'
fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
ax.plot(s_collection, label='At the beginning', color='y')
ax.plot(s11_collection, label='after 1k epochs', color='r')
ax.plot(s22_collection, label='after 2k epochs', color='g')
ax.plot(s33_collection, label='after 4k epochs', color='b')
ax.plot(s44_collection, label='after 8k epochs', color='k')
# ax.plot(yy, label='L2 error of predicted q-value', color='r')
# ax.plot(yym1, label='Avg. predicted q-value', color='g')
# ax.plot(yym2, label='Avg. ground truth q-value', color='b')
ax.set_xlabel('States')
ax.set_ylabel("K-cut value")
ax.set_title('state distribution in replay buffer during training')
plt.legend(loc="lower left")
plt.savefig(path + '/supervised/case_study/state_distr/' + fig_name + '.png')
plt.close()


n, bins, patches = plt.hist(x=s_collection, bins='auto', color='b',
                            alpha=0.7, rwidth=0.85)
n, bins, patches = plt.hist(x=s4_collection, bins='auto', color='r',
                            alpha=0.7, rwidth=0.85)
n, bins, patches = plt.hist(x=s44_collection, bins='auto', color='y',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('K-cut value')
plt.ylabel('Frequency')
plt.title('State distribution in replay buffer')
plt.legend(['beginning', 'dqn after 10k epochs', 'soft-dqn after 10k epochs'], loc='upper right')
plt.text(23, 45, r'$\mu=15, b=3$')
plt.savefig(path + '/supervised/case_study/state_distr/' + 'hist' + '.png')
plt.close()

x = []
for i in tqdm(range(alg.experience_replay_buffer.__len__())):
    # x.append(sum(alg.experience_replay_buffer[i].reward_seq))
    x.append(alg.experience_replay_buffer[i].r.item())
sum(x) / alg.replay_buffer_max_size

# 3 by 3
problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16, mode='complete', sample_episode=1)
test = test_summary(alg=alg, problem=problem, q_net='mlp')
test.run_test(trial_num=1, batch_size=1, gnn_step=3, episode_len=50, explore_prob=0.0, Temperature=0.05)
test.show_result()
# 5 by 6
problem = KCut_DGL(k=5, m=6, adjacent_reserve=10, hidden_dim=32, sample_episode=500)
test = test_summary(alg=alg, problem=problem, q_net='mlp')
test.run_test(trial_num=5, batch_size=100, gnn_step=3, episode_len=50, explore_prob=0.0, Temperature=0.05)
test.show_result()
# 10 by 10
problem = KCut_DGL(k=10, m=10, adjacent_reserve=20, hidden_dim=64, sample_episode=100)
test = test_summary(alg=alg, problem=problem, q_net='mlp')
test.run_test(trial_num=10, batch_size=10, gnn_step=3, episode_len=200, explore_prob=0.0, Temperature=0.05)
test.show_result()
# scp -r /u/fy4bc/code/research/RL4CombOptm/gnn_rl/Models/dqn_0113_test_eps0 fy4bc@128.143.69.125:/home/fy4bc/mnt/code/research/RL4CombOptm/MinimumVertexCover_DRL/Models/


# plot Q-loss/Reward curve
fig_name = 'return-base33-batchsize1000'

ret = alg.log.get_log("tot_return")
qv = alg.log.get_log("Q_error")
x = []
for i in range(len(qv)):
    if i*1+100<=len(ret):
        x.append(np.mean(ret[i*1:i*1+100]))

fig = plt.figure(figsize=[15, 5])
ax = fig.add_subplot(121)
ax.plot(x, label='episode reward')
ax2 = ax.twinx()
eps = np.concatenate([np.linspace(0.5, 0.1, 5000), np.ones(len(qv)-5000)*0.1]) #
# eps = np.linspace(0.9, 0.1, 1000)
ax2.plot(eps, label='\epsilon-greedy exploration prob.', color='r')
# fig.legend(loc=1)
ax.set_xlabel('Training Epochs')
ax.set_ylabel("Accumulated Episode Reward")
ax2.set_ylabel("Exploration Probability")
ax.set_title('Training Reward')
ax = fig.add_subplot(122)
ax.plot(qv[0:], label='episode reward')
ax2 = ax.twinx()
ax2.plot(eps, label='\epsilon-greedy exploration prob.', color='r')
# fig.legend(loc=1)
ax.set_xlabel('Training Epochs')
ax.set_ylabel("Qradratic Q-loss ")
ax2.set_ylabel("Exploration Probability")
ax.set_title('Training Loss')
plt.savefig('./Analysis/figs/' + fig_name + '.png')
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


path = '/p/reinforcement/data/gnn_rl/model/dqn/dqn_10by10_test_nan/'

with open(path + 'dqn_nan_debug', 'rb') as model_file:
    f = pickle.load(model_file)

# f = [self.model, sample_buffer, Q_s2a.view(-1, action_num).max(dim=1).values, Q_s1a_]

sample_buffer = f[1]

batch_size = 100
problem = KCut_DGL(k=10, m=10, adjacent_reserve=20, hidden_dim=64, sample_episode=100)

# make batches
batch_begin_state = dgl.batch([tpl.s0 for tpl in sample_buffer])
batch_end_state = dgl.batch([tpl.s1 for tpl in sample_buffer])
R = [tpl.r.unsqueeze(0) for tpl in sample_buffer]
batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
batch_end_action = problem.get_legal_actions(state=batch_end_state).cuda()
action_num = batch_end_action.shape[0] // batch_begin_action.shape[0]

g = dgl.unbatch(batch_begin_state)
h = torch.zeros((100, 64)).cuda()
h = f[0].layers[0](g[28], h)
h

# calculate edges
_, neighbor_idx, dist_matrix = dgl.transform.knn_graph(g[28].ndata['x'].view(1, 100, -1), 20 + 1, extend_info=True)
nonzero_idx = [i for i in range(100**2) if i % (100+1) != 0]
d = torch.sqrt(dist_matrix.reshape(batch_size, -1, 1)[:, nonzero_idx, :]).view(-1, 1)

a = torch.zeros((100, 100))
err = torch.zeros((100, 100))
for i in range(100):
    for j in range(100):
        a[i, j] = sum((g[28].ndata['x'][i] - g[28].ndata['x'][j])**2)
        if i != j:
            err[i, j] = (F.relu(dist_matrix[i, j]) - a[i, j]) / a[i, j]
