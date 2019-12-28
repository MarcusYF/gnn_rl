from DQN import DQN
from test import *
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch
import os
import gc
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
problem = KCut_DGL(k=5, m=6, adjacent_reserve=20, hidden_dim=16, random_init_label=True, a=1)
alg = DQN(problem, gamma=0.9, eps=0.1, lr=.02, replay_buffer_max_size=10, cuda_flag=True)

def run_dqn():
    for i in tqdm(range(50)):
        T1 = time.time()
        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log = alg.train_dqn(batch_size=128, num_episodes=1, episode_len=200, gcn_step=10, q_step=1, ddqn=True)
        T2 = time.time()
        # print('Epoch: {}. T: {}'.format(i, np.round(T2-T1,3)))
        print('Epoch: {}. R: {}. TD error: {}. H: {}. T: {}'.format(i, np.round(log.get_current('tot_return'),2),np.round(log.get_current('Q_error'),3),np.round(log.get_current('entropy'),3),np.round(T2-T1,3)))

run_dqn()