from DQN import DQN, to_cuda, EpisodeHistory
from k_cut import *
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch
import os
import pickle
from tqdm import tqdm
from toy_models.Qiter import vis_g

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
action_type = 'swap'
k = 3
m = 3
ajr = 5
hidden_dim = 16
extended_h = True
time_aware = False
a = 1
gamma = 0.90
lr = 1e-4
replay_buffer_max_size = 100 #
n_epoch = 10000
save_ckpt_step = 500
eps = np.linspace(0.5, 0.1, 5000) #
target_update_step = 5
batch_size = 1000
grad_accum = 1
sample_batch_episode = False
num_episodes = 10
episode_len = 50
gnn_step = 3
q_step = 1
ddqn = False

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
alg = DQN(problem, action_type=action_type
          , gamma=gamma, eps=.1, lr=lr
          , replay_buffer_max_size=replay_buffer_max_size
          , extended_h=extended_h
          , time_aware=time_aware
          , cuda_flag=True)

# path = 'Models/dqn_flip_test/'
path = 'Models/dqn_0114_base/'
if not os.path.exists(path):
    os.makedirs(path)
with open(path + 'dqn_0', 'wb') as model_file:
    pickle.dump(alg, model_file)


def run_dqn(alg):
    for i in tqdm(range(n_epoch)):

        if i % save_ckpt_step == save_ckpt_step - 1:
            with open(path + 'dqn_'+str(i+1), 'wb') as model_file:
                pickle.dump(alg, model_file)
            with open(path + 'dqn_'+str(i+1), 'rb') as model_file:
                alg = pickle.load(model_file)

        if i > len(eps) - 1:
            alg.eps = eps[-1]
        else:
            alg.eps = eps[i]

        T1 = time.time()
        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log = alg.train_dqn(sample_batch_episode=sample_batch_episode, batch_size=batch_size, grad_accum=grad_accum, num_episodes=num_episodes, episode_len=episode_len, gnn_step=gnn_step, q_step=q_step, ddqn=ddqn)
        if i % target_update_step == target_update_step - 1:
            alg.update_target_net()
        T2 = time.time()
        print('Epoch: {}. R: {}. Q error: {}. H: {}. T: {}'
              .format(i
               , np.round(log.get_current('tot_return'), 2)
               , np.round(log.get_current('Q_error'), 3)
               , np.round(log.get_current('entropy'), 3)
               , np.round(T2-T1, 3)))

run_dqn(alg)


#
#
# problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
# g = problem.g
# g.ndata['label']
# vis_g(problem, name='toy_models/a1', topo='c')
# S_a_encoding, h1, h2, Q_sa = alg.model(to_cuda(g), gnn_step=gnn_step, max_step=episode_len, remain_step=0)
# problem.reset_label([0,2,0,2,1,1,0,1,2])
# problem.calc_S()
#
# buf = alg.experience_replay_buffer[0]
# g_init = dc(buf.init_state)
# g_init.ndata['label'] # start
# g = dc(g_init)
#
# alg.problem.g.ndata['label'] # end
#
#
# g.ndata['label'] = g.ndata['label'][buf.label_perm[0], :]
# buf.action_seq = [(2, 7), (2, 8), (0, 1), (0, 2), (2, 6)]
# buf.action_indices = [tensor(14), tensor(15), tensor(0), tensor(1), tensor(13)]
# buf.reward_seq = tensor([ 0.4035, -0.4050,  0.5623, -0.2317,  0.3374])
#
# problem.reset_label([2,1,0,2,2,1,0,1,0])
# problem.g.ndata['label']
# #
# _, h1, h2, Q_sa1 = alg.model(to_cuda(problem.g), problem.get_legal_actions(), gnn_step=3)
# Q_sa1.argmax()
# state, reward = problem.step((4,8))
# _, h11, h22, Q_sa11 = alg.model(to_cuda(problem.g), problem.get_legal_actions(), gnn_step=3)
# Q_sa11.argmax()
#
# torch.norm(h2-h22, dim=1)
#
# a = alg.experience_replay_buffer[-1]
#
# _, _, _, Q_sa = alg.model(to_cuda(problem.g), problem.get_legal_actions()[1:2,:], gnn_step=3)
#
# d = {}
# for s in all_state:
#     print(s)
#     problem.reset_label(QtableKey2state(s))
#     act = problem.get_legal_actions()
#     _, _, _, Q_sa = alg.model(to_cuda(problem.g), problem.get_legal_actions(), gnn_step=3)
#     act_i = Q_sa.argmax()
#     d[s] = act[act_i]
#
# b = np.zeros([9, 9])
# for i in d.keys():
#     x = d[i][0]
#     y = d[i][1]
#     b[x,y] += 1



