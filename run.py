# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:30:53 2019

@author: fy4bc
"""

from DQN import DQN
from k_cut import *
import argparse
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch
import os
import pickle
from tqdm import tqdm
import json
from toy_models.Qiter import vis_g

def latestModelVersion(file):
    versions = [0]
    for root, dirs, files in os.walk(file):
        for f in files:
            if len(f.split('_')) > 1:
                versions.append(int(f.split('_')[1]))
    return max(versions)

# python run.py --gpu=0 --save_folder=dqn_0110_test_extend_h --extend_h=False --n_epoch=5000 --save_ckpt_step=500
# python run.py --gpu=1 --save_folder=dqn_0110_test_q_step2 --q_step=2 --n_epoch=5000 --save_ckpt_step=500
# python run.py --gpu=2 --save_folder=dqn_0110_test_gamma95 --gamma=0.95 --n_epoch=5000 --save_ckpt_step=500
# python run.py --gpu=3 --save_folder=dqn_0113_test_eps --eps=0.3
# args
parser = argparse.ArgumentParser(description="GNN with RL")
parser.add_argument('--save_folder', default='test')
parser.add_argument('--gpu', default='0', help="")
parser.add_argument('--resume', default=False)
parser.add_argument('--action_type', default='swap', help="")
parser.add_argument('--k', default=3, help="size of K-cut")
parser.add_argument('--m', default=3, help="cluster size")
parser.add_argument('--ajr', default=5, help="")
parser.add_argument('--h', default=16, help="hidden dimension")
parser.add_argument('--extend_h', default=True)
parser.add_argument('--time_aware', default=False)
parser.add_argument('--a', default=1, help="")
parser.add_argument('--gamma', type=float, default=0.9, help="")
parser.add_argument('--eps', type=float, default=0.1, help="")
parser.add_argument('--explore_end_at', type=float, default=0.2, help="")
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--n_epoch', default=5000)
parser.add_argument('--save_ckpt_step', default=500)
parser.add_argument('--target_update_step', default=5)
parser.add_argument('--replay_buffer_size', default=100, help="")
parser.add_argument('--batch_size', default=100, help='')
parser.add_argument('--grad_accum', default=10, help='')
parser.add_argument('--n_episode', default=10, help='')
parser.add_argument('--episode_len', default=50, help='')
parser.add_argument('--gnn_step', default=3, help='')
parser.add_argument('--q_step', default=1)
parser.add_argument('--ddqn', default=False)

args = vars(parser.parse_args())

save_folder = args['save_folder']
gpu = args['gpu']
resume = args['resume']
action_type = args['action_type']
k = int(args['k'])
m = int(args['m'])
ajr = int(args['ajr'])
h = int(args['h'])
extend_h = bool(args['extend_h'])
time_aware = bool(args['time_aware'])
a = int(args['a'])
gamma = float(args['gamma'])
lr = args['lr']    # learning rate
replay_buffer_size = int(args['replay_buffer_size'])
target_update_step = int(args['target_update_step'])
batch_size = int(args['batch_size'])
grad_accum = int(args['grad_accum'])
n_episode = int(args['n_episode'])
episode_len = int(args['episode_len'])
gnn_step = int(args['gnn_step'])
q_step = int(args['q_step'])
n_epoch = int(args['n_epoch'])
explore_end_at = float(args['explore_end_at'])
eps = np.linspace(1.0, float(args['eps']), int(n_epoch * explore_end_at))
save_ckpt_step = int(args['save_ckpt_step'])
ddqn = bool(args['ddqn'])

os.environ['CUDA_VISIBLE_DEVICES'] = gpu

# current working path
path = 'Models/' + save_folder + '/'
if not os.path.exists(path):
    os.makedirs(path)

# problem instances
problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h)

# model to be trained
if not resume:
    model_version = 0
    alg = DQN(problem, action_type=action_type
              , gamma=gamma, eps=.1, lr=lr
              , replay_buffer_max_size=replay_buffer_size
              , extended_h=extend_h
              , time_aware=time_aware
              , cuda_flag=True)
    with open(path + 'dqn_0', 'wb') as model_file:
        pickle.dump(alg, model_file)
else:
    model_version = latestModelVersion(path)
    with open(path + 'dqn_' + str(model_version), 'rb') as model_file:
        # might throw EOF error
        alg = pickle.load(model_file)

# record current training settings
if resume:
    mode = 'a+'
else:
    mode = 'w'
with open(path + 'params', mode) as params_file:
    params_file.write(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime()))
    params_file.write('\n------------------------------------\n')
    params_file.write(json.dumps(args))
    params_file.write('\n------------------------------------\n')



def run_dqn(alg):
    for i in tqdm(range(n_epoch)):

        if i % save_ckpt_step == save_ckpt_step - 1:
            with open(path + 'dqn_'+str(model_version+i+1), 'wb') as model_file:
                pickle.dump(alg, model_file)
            with open(path + 'dqn_'+str(model_version+i+1), 'rb') as model_file:
                alg = pickle.load(model_file)

        if i > len(eps) - 1:
            alg.eps = eps[-1]
        else:
            alg.eps = eps[i]

        T1 = time.time()
        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log = alg.train_dqn(batch_size=batch_size, grad_accum=grad_accum, num_episodes=n_episode, episode_len=episode_len, gnn_step=gnn_step, q_step=q_step, ddqn=ddqn)
        if i % target_update_step == target_update_step - 1:
            alg.update_target_net()
        T2 = time.time()
        print('Epoch: {}. R: {}. Q error: {}. H: {}. T: {}'
              .format(i
               , np.round(log.get_current('tot_return'), 2)
               # , log.get_current('R_signal')
               , np.round(log.get_current('Q_error'), 3)
               , np.round(log.get_current('entropy'), 3)
               , np.round(T2-T1, 3)))


if __name__ == '__main__':
    run_dqn(alg)
