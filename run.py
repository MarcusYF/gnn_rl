# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:30:53 2019

@author: fy4bc
"""

from DQN import DQN
from k_cut import *
import argparse
import numpy as np
import time
import os
from tqdm import tqdm

# args
parser = argparse.ArgumentParser(description="GNN with RL")
parser.add_argument('--gpu', default='0', help="")
parser.add_argument('--k', default=5, help="size of K-cut")
parser.add_argument('--m', default=6, help="cluster size")
parser.add_argument('--ajr', default=20, help="")
parser.add_argument('--h', default=16, help="hidden dimension")
parser.add_argument('--a', default=1, help="")
parser.add_argument('--gamma', type=float, default=0.9, help="")
parser.add_argument('--eps', type=float, default=0.1, help="")
parser.add_argument('--lr', type=float, default=0.02, help="learning rate")
parser.add_argument('--replay_buffer_size', default=10, help="")
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--grad_accum', default=10, help='')
parser.add_argument('--num_episode', default=1, help='')
parser.add_argument('--episode_len', default=500, help='')
parser.add_argument('--gcn_step', default=10, help='')
parser.add_argument('--q_step', default=1)
parser.add_argument('--num_epoch', default=50)
parser.add_argument('--target_update_step', default=5)
parser.add_argument('--ddqn', default=True)
args = vars(parser.parse_args())

gpu = args['gpu']
k = int(args['k'])
m = int(args['m'])
ajr = int(args['ajr'])
h = int(args['h'])
a = int(args['a'])
gamma = float(args['gamma'])
eps = float(args['eps'])
lr = args['lr']    # learning rate
replay_buffer_size = int(args['replay_buffer_size'])
B = int(args['batch_size'])
grad_accum = int(args['grad_accum'])
n_episode = int(args['num_episode'])
episode_len = int(args['episode_len'])
gcn_step = int(args['gcn_step'])
q_step = int(args['q_step'])
n_epoch = int(args['num_epoch'])
target_update_step = int(args['target_update_step'])
ddqn = bool(args['ddqn'])


os.environ['CUDA_VISIBLE_DEVICES'] = gpu

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, random_init_label=True, a=a)

alg = DQN(problem, gamma=gamma, eps=eps, lr=lr, replay_buffer_max_size=replay_buffer_size, cuda_flag=True)


def run_dqn():
    for i in tqdm(range(n_epoch)):
        T1 = time.time()
        log = alg.train_dqn(batch_size=B, grad_accum=grad_accum, num_episodes=n_episode, episode_len=episode_len, gcn_step=gcn_step, q_step=q_step, ddqn=ddqn)
        if i % target_update_step == target_update_step - 1:
            alg.update_target_net()
        T2 = time.time()
        print('Epoch: {}. R: {}. Q error: {}. H: {}. T: {}'.format(i, np.round(log.get_current('tot_return'),2),np.round(log.get_current('Q_error'),3),np.round(log.get_current('entropy'),3),np.round(T2-T1,3)))


if __name__ == '__main__':
    run_dqn()
