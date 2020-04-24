# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:30:53 2019

@author: fy4bc
"""
from networks import *
from envs import *
import argparse
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
import json
from validation import test_summary
from torch.utils.tensorboard import SummaryWriter

# server 2,3
model_folder = '/p/reinforcement/data/gnn_rl/model/dqn/'
log_folder = '/u/fy4bc/code/research/RL4CombOptm/gnn_rl/runs/'
# server 1
cuda_flag = True
test_seed0 = 1
model_folder = '/home/fy4bc/mnt/data/gnn_rl/model/dqn/'
log_folder = '/home/fy4bc/mnt/data/gnn_rl/logs/runs/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_folder = 'dsn_3by3_0423_4'
k = 3
m = 3
h = 16
batch_size = 1
test_episode = 1
n_epoch = 1000
lr = 1e-4
tau = 10
gnn_step = 3
graph_style = 'plain'

G = GraphGenerator(k=k, m=m, ajr=1, style=graph_style)
test_graphs = G.generate_graph(batch_size=test_episode, style=graph_style, seed=test_seed0, cuda_flag=cuda_flag)

# model to be trained
dsn = DSNet(k=k, hidden_dim=h, softmax_tau=tau, gnn_step=gnn_step)
if cuda_flag:
    dsn = dsn.cuda()



def run_dsn():


    writer = SummaryWriter(log_folder + save_folder)
    optimizer = torch.optim.Adam(dsn.parameters(), lr=lr)

    for n_iter in tqdm(range(n_epoch)):

        t1 = time.time()
        train_batch = G.generate_graph(batch_size=batch_size, style=graph_style, seed=test_seed0, cuda_flag=cuda_flag)
        t2 = time.time()
        kcut_S = dsn(train_batch)
        L = torch.mean(kcut_S)
        t3 = time.time()
        L.backward()
        optimizer.step()
        optimizer.zero_grad()
        t4 = time.time()

        print('Epoch: {}. Loss: {}. T1: {}. T2: {}. T3: {}'
              .format(n_iter
               , np.round(L.item(), 2)
               , np.round(t2 - t1, 3), np.round(t3 - t2, 3), np.round(t4 - t3, 3)))


        if n_iter % 100 == 0:
            validation_loss = dsn(test_graphs, tau=None).mean().item()

            writer.add_scalar('Loss/Q-Loss', L.item(), n_iter)
            writer.add_scalar('Reward/Validation Episode Reward - easy', validation_loss, n_iter)

            writer.add_scalar('Time/Running Time per Epoch', t4 - t1, n_iter)

if __name__ == '__main__':

    run_dsn()
