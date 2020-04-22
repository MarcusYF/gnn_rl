# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:30:53 2019

@author: fy4bc
"""

from ddqn import DQN
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

print('new job..')
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

# server 2,3
model_folder = '/p/reinforcement/data/gnn_rl/model/dqn/'
log_folder = '/u/fy4bc/code/research/RL4CombOptm/gnn_rl/runs/'
# server 1
test_seed0, test_seed1, test_seed2 = 1, 11, 111
model_folder = '/home/fy4bc/mnt/data/gnn_rl/model/dqn/'
log_folder = '/home/fy4bc/mnt/data/gnn_rl/logs/runs/'

parser = argparse.ArgumentParser(description="GNN with RL")
parser.add_argument('--save_folder', default='dqn_3by3_0421_1')
parser.add_argument('--train_distr', default='plain', help="")
parser.add_argument('--test_distr0', default='plain', help="")
parser.add_argument('--target_mode', default=False)
parser.add_argument('--test_distr1', default='plain', help="")
parser.add_argument('--test_distr2', default='plain', help="")
parser.add_argument('--k', default=3, help="size of K-cut")
parser.add_argument('--m', default='3', help="cluster size")
parser.add_argument('--ajr', default=8, help="")
parser.add_argument('--h', default=32, help="hidden dimension")
parser.add_argument('--rollout_step', default=0)
parser.add_argument('--q_step', default=1)
parser.add_argument('--batch_size', default=500, help='')
parser.add_argument('--n_episode', default=1, help='')
parser.add_argument('--episode_len', default=50, help='')
parser.add_argument('--action_type', default='swap', help="")
parser.add_argument('--gnn_step', default=3, help='')

parser.add_argument('--gpu', default='0', help="")
parser.add_argument('--resume', default=False)
parser.add_argument('--problem_mode', default='complete', help="")
parser.add_argument('--readout', default='mlp', help="")
parser.add_argument('--edge_info', default='adj_dist')
parser.add_argument('--clip_target', default=0)
parser.add_argument('--explore_method', default='epsilon_greedy')
parser.add_argument('--priority_sampling', default=0)
parser.add_argument('--gamma', type=float, default=0.9, help="")
parser.add_argument('--eps0', type=float, default=0.5, help="")
parser.add_argument('--eps', type=float, default=0.1, help="")
parser.add_argument('--explore_end_at', type=float, default=0.9, help="")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--action_dropout', type=float, default=1.0)
parser.add_argument('--n_epoch', default=30000)
parser.add_argument('--save_ckpt_step', default=30000)
parser.add_argument('--target_update_step', default=5)
parser.add_argument('--replay_buffer_size', default=5000, help="")
parser.add_argument('--test_batch_size', default=100, help='')
parser.add_argument('--grad_accum', default=1, help='')
parser.add_argument('--sample_batch_episode', type=int, default=0, help='')  # 0 for cascade sampling and 1 for batch episode sampling
parser.add_argument('--ddqn', default=False)

args = vars(parser.parse_args())
save_folder = args['save_folder']
gpu = args['gpu']
resume = args['resume']
target_mode = args['target_mode']
problem_mode = args['problem_mode']
readout = args['readout']
action_type = args['action_type']
k = int(args['k'])
m = [int(i) for i in args['m'].split(',')]
if len(m) == 1:
    m = m[0]
    N = k * m
else:
    N = sum(m)
if k == 3 and m == 4:
    run_validation_33 = True
else:
    run_validation_33 = False
ajr = int(args['ajr'])
train_graph_style = args['train_distr']
test_graph_style_0 = args['test_distr0']
test_graph_style_1 = args['test_distr1']
test_graph_style_2 = args['test_distr2']
h = int(args['h'])
edge_info = args['edge_info']
clip_target = bool(int(args['clip_target']))
explore_method = args['explore_method']
priority_sampling = bool(int(args['priority_sampling']))
gamma = float(args['gamma'])
lr = args['lr']    # learning rate
action_dropout = args['action_dropout']    # learning rate
replay_buffer_size = int(args['replay_buffer_size'])
target_update_step = int(args['target_update_step'])
batch_size = int(args['batch_size'])
grad_accum = int(args['grad_accum'])
sample_batch_episode = bool(args['sample_batch_episode'])
n_episode = int(args['n_episode'])
test_episode = int(args['test_batch_size'])
episode_len = int(args['episode_len'])
gnn_step = int(args['gnn_step'])
rollout_step = int(args['rollout_step'])
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
path = model_folder + save_folder + '/'
if not os.path.exists(path):
    os.makedirs(path)


G = GraphGenerator(k=k, m=m, ajr=ajr, style=train_graph_style)
test_problem0 = G
test_problem1 = G
test_problem2 = G

# model to be trained
alg = DQN(graph_generator=G, hidden_dim=h, action_type=action_type
              , gamma=gamma, eps=.1, lr=lr, action_dropout=action_dropout
              , sample_batch_episode=sample_batch_episode
              , replay_buffer_max_size=replay_buffer_size
              , epi_len=episode_len
              , new_epi_batch_size=n_episode
              , cuda_flag=True
              , explore_method=explore_method
              , priority_sampling=priority_sampling)
if not resume:
    model_version = 0
    with open(path + 'dqn_0', 'wb') as model_file:
        pickle.dump(alg, model_file)
else:
    model_version = latestModelVersion(path)
    with open(path + 'dqn_' + str(model_version), 'rb') as model_file:
        # might throw EOF error
        model = pickle.load(model_file)
    alg.model = model
    alg.model_target = model


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


bg_easy = test_problem0.generate_graph(batch_size=test_episode, style=test_graph_style_0, seed=test_seed0)
bg_hard = test_problem1.generate_graph(batch_size=test_episode, style=test_graph_style_1, seed=test_seed1)
bg_subopt = test_problem2.generate_graph(batch_size=test_episode, style=test_graph_style_2, seed=test_seed2)

print('bg_easy.ndata[x][0]', bg_easy.ndata['x'][0])

target_bg = None
if target_mode:
    if bg_easy.ndata.keys().__contains__('h'):
        bg_easy.ndata.pop('h')
    target_bg = dgl.unbatch(bg_easy)


def run_dqn(alg):

    t = 0
    writer = SummaryWriter(log_folder + save_folder)
    for n_iter in tqdm(range(n_epoch)):

        T1 = time.time()

        if n_iter > len(eps) - 1:
            alg.eps = eps[-1]
        else:
            alg.eps = eps[n_iter]

        T11 = time.time()
        # TODO memory usage :: episode_len * num_episodes * hidden_dim
        log = alg.train_dqn(target_bg=target_bg
                            , epoch=n_iter
                            , batch_size=batch_size
                            , num_episodes=n_episode
                            , episode_len=episode_len
                            , gnn_step=gnn_step
                            , q_step=q_step
                            , rollout_step=rollout_step
                            , ddqn=ddqn)
        if n_iter % target_update_step == target_update_step - 1:
            alg.update_target_net()
        T22 = time.time()
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
            t += 1
            # with open(path + 'buffer_' + str(model_version + n_iter + 1), 'wb') as model_file:
            #     pickle.dump([alg.cascade_replay_buffer_weight, [[problem.calc_S(g=elem.s0) for elem in alg.cascade_replay_buffer[i]] for i in range(len(alg.cascade_replay_buffer))]]
            #                  , model_file)

        # validation

        # test summary
        if n_iter % 100 == 0:
            test = test_summary(alg=alg, problem=test_problem1, action_type=action_type, q_net=readout, forbid_revisit=0)

            test.run_test(problem=bg_hard, trial_num=1, batch_size=100, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
            epi_r0 = test.show_result()


            test.problem = test_problem0
            test.run_test(problem=bg_easy, trial_num=1, batch_size=test_episode, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
            epi_r1 = test.show_result()


            test.problem = test_problem2
            test.run_test(problem=bg_subopt, trial_num=1, batch_size=100, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
            epi_r2 = test.show_result()


        writer.add_scalar('Reward/Training Episode Reward', log.get_current('tot_return') / n_episode, n_iter)
        writer.add_scalar('Loss/Q-Loss', log.get_current('Q_error'), n_iter)
        writer.add_scalar('Reward/Validation Episode Reward - hard', epi_r0, n_iter)
        writer.add_scalar('Reward/Validation Episode Reward - easy', epi_r1, n_iter)
        writer.add_scalar('Reward/Validation Episode Reward - subopt', epi_r2, n_iter)
        writer.add_scalar('Time/Running Time per Epoch', T2 - T1, n_iter)

if __name__ == '__main__':

    run_dqn(alg)
