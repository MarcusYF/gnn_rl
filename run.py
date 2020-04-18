# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:30:53 2019

@author: fy4bc
"""

from DQN import DQN, to_cuda
from k_cut import *
import argparse
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
import json
from Analysis.episode_stats import test_summary
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
model_folder = '/home/fy4bc/mnt/data/gnn_rl/model/dqn/'
log_folder = '/home/fy4bc/mnt/data/gnn_rl/logs/runs/'

parser = argparse.ArgumentParser(description="GNN with RL")
parser.add_argument('--save_folder', default='dqn_4by4_0418_1')
parser.add_argument('--train_distr', default='plain', help="")
parser.add_argument('--test_distr0', default='plain', help="")
parser.add_argument('--target_mode', default=False)
parser.add_argument('--test_distr1', default='plain', help="")
parser.add_argument('--test_distr2', default='plain', help="")
parser.add_argument('--k', default=4, help="size of K-cut")
parser.add_argument('--m', default='4', help="cluster size")
parser.add_argument('--ajr', default=15, help="")
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
parser.add_argument('--eps0', type=float, default=0.8, help="")
parser.add_argument('--eps', type=float, default=0.1, help="")
parser.add_argument('--explore_end_at', type=float, default=0.3, help="")
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
if k == 3 and m == 3:
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
# eps = np.exp( np.linspace( np.log( float(args['eps0']) ), np.log( float(args['eps'])), int(n_epoch * explore_end_at)) )
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

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, mode=problem_mode, sample_episode=n_episode, graph_style=train_graph_style)
# test_problem0 = KCut_DGL(k=k, m=4, adjacent_reserve=15, hidden_dim=h, mode=problem_mode, sample_episode=test_episode)
# test_problem1 = KCut_DGL(k=k, m=4+1, adjacent_reserve=19, hidden_dim=h, mode=problem_mode, sample_episode=test_episode)
# test_problem2 = KCut_DGL(k=k, m=4+5, adjacent_reserve=23, hidden_dim=h, mode=problem_mode, sample_episode=test_episode)
test_problem0 = problem
test_problem1 = problem
test_problem2 = problem

# model to be trained
alg = DQN(problem, action_type=action_type
              , gamma=gamma, eps=.1, lr=lr, action_dropout=action_dropout
              , sample_batch_episode=sample_batch_episode
              , replay_buffer_max_size=replay_buffer_size
              , epi_len=episode_len
              , new_epi_batch_size=n_episode
              , edge_info=edge_info
              , readout=readout
              , explore_method=explore_method
              , priority_sampling=priority_sampling
              , clip_target=clip_target)
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


# load validation set
if run_validation_33:
    with open('/p/reinforcement/data/gnn_rl/model/test_data/3by3/0', 'rb') as valid_file:
        validation_problem0 = pickle.load(valid_file)
    with open('/p/reinforcement/data/gnn_rl/model/test_data/3by3/1', 'rb') as valid_file:
        validation_problem1 = pickle.load(valid_file)
    bg_hard = to_cuda(dgl.batch([p[0].g for p in validation_problem0[:test_episode]]))
    bg_easy = to_cuda(dgl.batch([p[0].g for p in validation_problem1[:test_episode]]))

    bg_subopt = []
    for i in range(test_episode):
        gi = to_cuda(validation_problem0[:test_episode][i][0].g)
        problem.reset_label(g=gi, label=validation_problem0[:test_episode][i][2])
        bg_subopt.append(gi)
    bg_subopt = dgl.batch(bg_subopt)

    for bg_ in [bg_hard, bg_easy, bg_subopt]:
        if ajr == 8:
            bg_.edata['e_type'][:, 0] = torch.ones(N * ajr * bg_.batch_size)
        _, _, square_dist_matrix = dgl.transform.knn_graph(bg_.ndata['x'].view(test_episode, N, -1), ajr+1, extend_info=True)
        square_dist_matrix = F.relu(square_dist_matrix, inplace=True)  # numerical error could result in NaN in sqrt. value
        bg_.ndata['adj'] = torch.sqrt(square_dist_matrix).view(bg_.number_of_nodes(), -1)
else:
    bg_easy = to_cuda(test_problem0.gen_batch_graph(batch_size=test_episode, style=test_graph_style_0))
    bg_hard = to_cuda(test_problem1.gen_batch_graph(batch_size=test_episode, style=test_graph_style_1))
    bg_subopt = to_cuda(test_problem2.gen_batch_graph(batch_size=test_episode, style=test_graph_style_2))

    # bg_easy = to_cuda(problem.gen_batch_graph(batch_size=test_episode, style=test_graph_style_0))  # 1
    # bg_hard = to_cuda(problem.gen_batch_graph(batch_size=test_episode, style=test_graph_style_1))  # 0
    # bg_subopt = to_cuda(problem.gen_batch_graph(batch_size=test_episode, style=test_graph_style_2))  # 2

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

        # if n_iter < 2500:
        #     alg.gamma = 0
        # elif n_iter < 5000:
        #     alg.gamma = 0.3
        # elif n_iter < 7500:
        #     alg.gamma = 0.5
        # elif n_iter < 10000:
        #     alg.gamma = 0.7
        # elif n_iter < 15000:
        #     alg.gamma = 0.9

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

            test.run_test(problem=to_cuda(bg_hard), trial_num=1, batch_size=100, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
            epi_r0 = test.show_result()
            if run_validation_33:
                best_hit0 = test.compare_opt(validation_problem0)

            test.problem = test_problem0
            test.run_test(problem=to_cuda(bg_easy), trial_num=1, batch_size=test_episode, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
            epi_r1 = test.show_result()
            if run_validation_33:
                best_hit1 = test.compare_opt(validation_problem1)

            test.problem = test_problem2
            test.run_test(problem=to_cuda(bg_subopt), trial_num=1, batch_size=100, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
            epi_r2 = test.show_result()
            if run_validation_33:
                best_hit2 = test.compare_opt(validation_problem0)

        writer.add_scalar('Reward/Training Episode Reward', log.get_current('tot_return') / n_episode, n_iter)
        writer.add_scalar('Loss/Q-Loss', log.get_current('Q_error'), n_iter)
        writer.add_scalar('Reward/Validation Episode Reward - hard', epi_r0, n_iter)
        writer.add_scalar('Reward/Validation Episode Reward - easy', epi_r1, n_iter)
        writer.add_scalar('Reward/Validation Episode Reward - subopt', epi_r2, n_iter)
        if run_validation_33:
            writer.add_scalar('Reward/Validation Opt. hit percent - hard', best_hit0, n_iter)
            writer.add_scalar('Reward/Validation Opt. hit percent - easy', best_hit1, n_iter)
            writer.add_scalar('Reward/Validation Opt. hit percent - subopt', best_hit2, n_iter)
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
# import matplotlib.pyplot as plt
# from k_cut import *
#
#
#
# test_problem0 = KCut_DGL(k=4, m=5, adjacent_reserve=19, hidden_dim=1)
# bg_easy = to_cuda(test_problem0.gen_batch_graph(batch_size=1, style='cluster-2'))
# problem=bg_easy
# k=4
# topo='cut'
# name='/u/fy4bc/u/fy4bc/code/research/RL4CombOptm/gnn_rl/toy_models/figs/test1'
#
# plt.figure(figsize=(5,5))
# if isinstance(problem, dgl.DGLGraph):
#     g = problem
#     k = k
# else:
#     k = problem.k
#     g = problem.g
# X = g.ndata['x'].cpu()
# n = X.shape[0]
# label = g.ndata['label'].cpu()
# link = dc(g.edata['e_type'].view(n, n - 1, 2).cpu())
# # c = ['r', 'b', 'y']
# plt.cla()
# c = ['r', 'b', 'y', 'k', 'g', 'c', 'm', 'tan', 'peru', 'pink']
# for i in range(k):
#     a = X[(label[:, i] > 0).nonzero().squeeze()]
#     if a.shape[0] > .5:
#         a = a.view(-1, 2)
#         plt.scatter(a[:, 0], a[:, 1], s=60, c=[c[i]]*(n//k))
#
# for i in range(n):
#     plt.annotate(str(i), xy=(X[i, 0], X[i, 1]))
#     for j in range(n - 1):
#         if link[i, j][0].item() == 1:
#             j_ = j
#             if j >= i:
#                 j_ = j + 1
#             # plt.plot([X[i, 0], X[j_, 0]], [X[i, 1], X[j_, 1]], ':', color='k')
#
#     if topo == 'cut':
#         for j in range(n - 1):
#             if link[i, j][1].item() + link[i, j][0].item() > 1.5:
#                 j_ = j
#                 if j >= i:
#                     j_ = j + 1
#                 # plt.plot([X[i, 0], X[j_, 0]], [X[i, 1], X[j_, 1]], '-', color='k')
#
# plt.savefig(name + '.png')
# plt.close()
# with open('/p/reinforcement/data/gnn_rl/model/test_data/3by3/1', 'rb') as valid_file:
#     validation_problem1 = pickle.load(valid_file)
# problem = KCut_DGL(k=3, m=3, adjacent_reserve=8, hidden_dim=1)
# a = 0
# b = 0
# c = 0
# for i in range(100):
#     problem.g = dc(validation_problem1[i][0].g)
#     a+=problem.calc_S()
#     problem.reset_label(label=validation_problem1[i][1])
#     b+=problem.calc_S()
#     problem.reset_label(label=validation_problem1[i][2])
#     c+=problem.calc_S()

