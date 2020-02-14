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
import pickle
from tqdm import tqdm
import json
import math
from torch.utils.tensorboard import SummaryWriter

# ensure reproducibility
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(1)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def latestModelVersion(file):
    versions = [0]
    for root, dirs, files in os.walk(file):
        for f in files:
            if len(f.split('_')) > 1:
                versions.append(int(f.split('_')[-1]))
    return max(versions)

# python run.py --gpu=0 --save_folder=dqn_0110_test_extend_h --extend_h=False --n_epoch=5000 --save_ckpt_step=500
# python run.py --gpu=1 --save_folder=dqn_0110_test_q_step2 --q_step=2 --n_epoch=5000 --save_ckpt_step=500
# python run.py --gpu=2 --save_folder=dqn_0110_test_gamma95 --gamma=0.95 --n_epoch=5000 --save_ckpt_step=500
# python run.py --gpu=1 --save_folder=dqn_0113_test_eps1 --explore_end_at=0.6 --eps=0.1
# python run.py --gpu=0 --save_folder=dqn_0113_test_eps0 --eps=0.3
# python run.py --gpu=1 --save_folder=dqn_0124_test_fix_target_0
# args
parser = argparse.ArgumentParser(description="GNN with RL")
parser.add_argument('--save_folder', default='dqn_5by6_0213_base')
parser.add_argument('--gpu', default='0', help="")
parser.add_argument('--resume', default=False)
parser.add_argument('--problem_mode', default='complete', help="")
parser.add_argument('--readout', default='mlp', help="")
parser.add_argument('--action_type', default='swap', help="")
parser.add_argument('--k', default=5, help="size of K-cut")
parser.add_argument('--m', default=6, help="cluster size")
parser.add_argument('--ajr', default=10, help="")
parser.add_argument('--h', default=32, help="hidden dimension")
parser.add_argument('--extend_h', default=True)
parser.add_argument('--use_x', default=0)
parser.add_argument('--edge_info', default='adj_dist')
parser.add_argument('--clip_target', default=0)
parser.add_argument('--explore_method', default='soft_dqn')
parser.add_argument('--priority_sampling', default=0)
parser.add_argument('--time_aware', default=False)
parser.add_argument('--a', default=1, help="")
parser.add_argument('--gamma', type=float, default=0.9, help="")
parser.add_argument('--eps0', type=float, default=0.2, help="")
parser.add_argument('--eps', type=float, default=0.05, help="")
parser.add_argument('--explore_end_at', type=float, default=0.5, help="")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--n_epoch', default=50000)
parser.add_argument('--save_ckpt_step', default=10000)
parser.add_argument('--target_update_step', default=5)
parser.add_argument('--replay_buffer_size', default=5000, help="")
parser.add_argument('--batch_size', default=200, help='')
parser.add_argument('--grad_accum', default=1, help='')
parser.add_argument('--sample_batch_episode', type=int, default=0, help='')  # 0 for cascade sampling and 1 for batch episode sampling
parser.add_argument('--n_episode', default=4, help='')
parser.add_argument('--episode_len', default=50, help='')
parser.add_argument('--gnn_step', default=3, help='')
parser.add_argument('--q_step', default=1)
parser.add_argument('--ddqn', default=False)

args = vars(parser.parse_args())
save_folder = args['save_folder']
gpu = args['gpu']
resume = args['resume']
problem_mode = args['problem_mode']
readout = args['readout']
action_type = args['action_type']
k = int(args['k'])
m = int(args['m'])
ajr = int(args['ajr'])
h = int(args['h'])
extend_h = bool(args['extend_h'])
use_x = bool(int(args['use_x']))
edge_info = args['edge_info']
clip_target = bool(int(args['clip_target']))
explore_method = args['explore_method']
priority_sampling = bool(int(args['priority_sampling']))
time_aware = bool(args['time_aware'])
a = int(args['a'])
gamma = float(args['gamma'])
lr = args['lr']    # learning rate
replay_buffer_size = int(args['replay_buffer_size'])
target_update_step = int(args['target_update_step'])
batch_size = int(args['batch_size'])
grad_accum = int(args['grad_accum'])
sample_batch_episode = bool(args['sample_batch_episode'])
n_episode = int(args['n_episode'])
episode_len = int(args['episode_len'])
gnn_step = int(args['gnn_step'])
q_step = int(args['q_step'])
n_epoch = int(args['n_epoch'])
explore_end_at = float(args['explore_end_at'])
eps = np.linspace(float(args['eps0']), float(args['eps']), int(n_epoch * explore_end_at))
save_ckpt_step = int(args['save_ckpt_step'])
ddqn = bool(args['ddqn'])

os.environ['CUDA_VISIBLE_DEVICES'] = gpu

# current working path
# absroot = os.path.dirname(os.getcwd())
# path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/Models/' + save_folder + '/'
path = '/p/reinforcement/data/gnn_rl/model/dqn/' + save_folder + '/'
if not os.path.exists(path):
    os.makedirs(path)


problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, mode=problem_mode, sample_episode=n_episode)

# model to be trained
if not resume:
    model_version = 0
    alg = DQN(problem, action_type=action_type
              , gamma=gamma, eps=.1, lr=lr
              , sample_batch_episode=sample_batch_episode
              , replay_buffer_max_size=replay_buffer_size
              , epi_len=episode_len
              , new_epi_batch_size=n_episode
              , extended_h=extend_h
              , time_aware=time_aware
              , use_x=use_x
              , edge_info=edge_info
              , readout=readout
              , explore_method=explore_method
              , priority_sampling=priority_sampling
              , clip_target=clip_target)
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
    # Gamma = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    t = 0
    writer = SummaryWriter('runs/' + save_folder)
    for n_iter in tqdm(range(n_epoch)):

        T1 = time.time()

        if n_iter > len(eps) - 1:
            alg.eps = eps[-1]
        else:
            alg.eps = eps[n_iter]

        T11 = time.time()
        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log, sample_buffer = alg.train_dqn(epoch=n_iter, batch_size=batch_size, num_episodes=n_episode, episode_len=episode_len, gnn_step=gnn_step, q_step=q_step, ddqn=ddqn)
        if n_iter % target_update_step == target_update_step - 1:
            alg.update_target_net()
        T22 = time.time()
        print(alg.gamma)
        print('Epoch: {}. R: {}. Q error: {}. H: {}. T: {}'
              .format(n_iter
               , np.round(log.get_current('tot_return'), 2)
               , np.round(log.get_current('Q_error'), 3)
               , np.round(log.get_current('entropy'), 3)
               , np.round(T22 - T11, 3)))

        T2 = time.time()

        if n_iter % save_ckpt_step == save_ckpt_step - 1:
            with open(path + 'dqn_'+str(model_version + n_iter + 1), 'wb') as model_file:
                pickle.dump(alg.model, model_file)
            # alg.gamma = Gamma[t]
            t += 1
                # pickle.dump([alg.model
                #             , alg.log.get_log("tot_return")
                #             , alg.log.get_log("Q_error")
                #             , alg.cascade_replay_buffer_weight]
                #             , model_file)
            with open(path + 'buffer_' + str(model_version + n_iter + 1), 'wb') as model_file:
                pickle.dump([alg.cascade_replay_buffer_weight, [[problem.calc_S(g=elem.s0) for elem in alg.cascade_replay_buffer[i]] for i in range(len(alg.cascade_replay_buffer))]]
                             , model_file)

        writer.add_scalar('Reward/Training Episode Reward', log.get_current('tot_return') / n_episode, n_iter)
        writer.add_scalar('Loss/Q-Loss', log.get_current('Q_error'), n_iter)
        writer.add_scalar('Time/Running Time per Epoch', T2 - T1, n_iter)

if __name__ == '__main__':
    run_dqn(alg)

# sample_buffer = alg[4]
# batch_begin_state = dgl.batch([tpl.s0 for tpl in sample_buffer])
# batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
#
# problem = KCut_DGL(k=10, m=10, adjacent_reserve=20, hidden_dim=64)
#
# batch_begin_state1 = problem.gen_batch_graph(batch_size=1, hidden_dim=64)
# batch_begin_action1 = problem.get_legal_actions(batch_begin_state1).cuda()
#
# _,_,_,Q_sa=alg[0].forward(to_cuda(batch_begin_state1), batch_begin_action1[0].unsqueeze(0))
#




