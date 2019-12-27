# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:30:53 2019

@author: fy4bc
"""

from DQN import DQN
from test import *
import matplotlib.pyplot as plt
from smooth_signal import smooth
import numpy as np
import time
import torch
import os
import gc
from memory_profiler import profile
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


problem = KCut_DGL(k=5, m=6, adjacent_reserve=20, hidden_dim=16, random_init_label=True, a=1)

alg = DQN(problem, gamma=0.9, eps=0.1, lr=.02, cuda_flag=True)


@profile
def run_dqn():
    for i in range(1500):
        T1 = time.time()
        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log = alg.train_dqn_test_mem(batch_size=16, num_episodes=5, episode_len=100, gcn_step=10, q_step=1)
        T2 = time.time()
        print('Epoch: {}. T: {}'.format(i, np.round(T2-T1,3)))
        # print('Epoch: {}. R: {}. TD error: {}. H: {}. T: {}'.format(i, np.round(log.get_current('tot_return'),2),np.round(log.get_current('TD_error'),3),np.round(log.get_current('entropy'),3),np.round(T2-T1,3)))

# if __name__ == '__main__':
run_dqn()