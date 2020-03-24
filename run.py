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

# python run.py --gpu=0 --save_folder=dqn_0110_test_extend_h --extend_h=False --n_epoch=5000 --save_ckpt_step=500
# python run.py --gpu=1 --save_folder=dqn_0110_test_q_step2 --q_step=2 --n_epoch=5000 --save_ckpt_step=500
# python run.py --gpu=2 --save_folder=dqn_0110_test_gamma95 --gamma=0.95 --n_epoch=5000 --save_ckpt_step=500
# python run.py --gpu=1 --save_folder=dqn_0113_test_eps1 --explore_end_at=0.6 --eps=0.1
# python run.py --gpu=0 --save_folder=dqn_0113_test_eps0 --eps=0.3
# python run.py --gpu=1 --save_folder=dqn_0124_test_fix_target_0
# args
parser = argparse.ArgumentParser(description="GNN with RL")
parser.add_argument('--save_folder', default='dqn_5by6_0323_plain')
parser.add_argument('--gpu', default='0', help="")
parser.add_argument('--resume', default=False)
parser.add_argument('--problem_mode', default='complete', help="")
parser.add_argument('--readout', default='mlp', help="")
parser.add_argument('--action_type', default='swap', help="")
parser.add_argument('--k', default=5, help="size of K-cut")
parser.add_argument('--m', default=6, help="cluster size")
parser.add_argument('--ajr', default=29, help="")
parser.add_argument('--style', default='plain', help="")
parser.add_argument('--h', default=32, help="hidden dimension")
parser.add_argument('--extend_h', default=True)
parser.add_argument('--use_x', default=0)
parser.add_argument('--edge_info', default='adj_dist')
parser.add_argument('--clip_target', default=0)
parser.add_argument('--explore_method', default='epsilon_greedy')
parser.add_argument('--priority_sampling', default=0)
parser.add_argument('--time_aware', default=False)
parser.add_argument('--a', default=1, help="")
parser.add_argument('--gamma', type=float, default=0.90, help="")
parser.add_argument('--eps0', type=float, default=0.5, help="")
parser.add_argument('--eps', type=float, default=0.1, help="")
parser.add_argument('--explore_end_at', type=float, default=0.2, help="")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--action_dropout', type=float, default=1.0)
parser.add_argument('--n_epoch', default=50000)
parser.add_argument('--save_ckpt_step', default=10000)
parser.add_argument('--target_update_step', default=5)
parser.add_argument('--replay_buffer_size', default=5000, help="")
parser.add_argument('--batch_size', default=500, help='')
parser.add_argument('--grad_accum', default=1, help='')
parser.add_argument('--sample_batch_episode', type=int, default=0, help='')  # 0 for cascade sampling and 1 for batch episode sampling
parser.add_argument('--n_episode', default=10, help='')
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
graph_style = args['style']
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
action_dropout = args['action_dropout']    # learning rate
replay_buffer_size = int(args['replay_buffer_size'])
target_update_step = int(args['target_update_step'])
batch_size = int(args['batch_size'])
grad_accum = int(args['grad_accum'])
sample_batch_episode = bool(args['sample_batch_episode'])
n_episode = int(args['n_episode'])
test_episode = 100
episode_len = int(args['episode_len'])
gnn_step = int(args['gnn_step'])
q_step = int(args['q_step'])
n_epoch = int(args['n_epoch'])
explore_end_at = float(args['explore_end_at'])
eps = np.linspace(float(args['eps0']), float(args['eps']), int(n_epoch * explore_end_at))
save_ckpt_step = int(args['save_ckpt_step'])
ddqn = bool(args['ddqn'])

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
run_validation = False

# current working path
# absroot = os.path.dirname(os.getcwd())
# path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/Models/' + save_folder + '/'
path = '/p/reinforcement/data/gnn_rl/model/dqn/' + save_folder + '/'
if not os.path.exists(path):
    os.makedirs(path)


problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, mode=problem_mode, sample_episode=n_episode, graph_style=graph_style)
test_problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, mode=problem_mode, sample_episode=test_episode)

# model to be trained
alg = DQN(problem, action_type=action_type
              , gamma=gamma, eps=.1, lr=lr, action_dropout=action_dropout
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
if run_validation:
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

    if ajr == 8:
        bg_hard.edata['e_type'][:, 0] = torch.ones(k * m * ajr * bg_hard.batch_size)
        bg_easy.edata['e_type'][:, 0] = torch.ones(k * m * ajr * bg_easy.batch_size)
        bg_subopt.edata['e_type'][:, 0] = torch.ones(k * m * ajr * bg_subopt.batch_size)


def run_dqn(alg):

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
        log = alg.train_dqn(epoch=n_iter, batch_size=batch_size, num_episodes=n_episode, episode_len=episode_len, gnn_step=gnn_step, q_step=q_step, ddqn=ddqn)
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
            t += 1
            # with open(path + 'buffer_' + str(model_version + n_iter + 1), 'wb') as model_file:
            #     pickle.dump([alg.cascade_replay_buffer_weight, [[problem.calc_S(g=elem.s0) for elem in alg.cascade_replay_buffer[i]] for i in range(len(alg.cascade_replay_buffer))]]
            #                  , model_file)

        # validation

        # test summary
        if n_iter % 100 == 0 and run_validation:
            test = test_summary(alg=alg, problem=test_problem, q_net=readout, forbid_revisit=0)

            test.run_test(problem=to_cuda(bg_hard), trial_num=1, batch_size=100, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
            epi_r0 = test.show_result()
            best_hit0 = test.compare_opt(validation_problem0)

            test.run_test(problem=to_cuda(bg_easy), trial_num=1, batch_size=100, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
            epi_r1 = test.show_result()
            best_hit1 = test.compare_opt(validation_problem1)

            test.run_test(problem=to_cuda(bg_subopt), trial_num=1, batch_size=100, gnn_step=gnn_step,
                          episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
            epi_r2 = test.show_result()
            best_hit2 = test.compare_opt(validation_problem0)

        writer.add_scalar('Reward/Training Episode Reward', log.get_current('tot_return') / n_episode, n_iter)
        writer.add_scalar('Loss/Q-Loss', log.get_current('Q_error'), n_iter)
        if run_validation:
            writer.add_scalar('Reward/Validation Episode Reward - hard', epi_r0, n_iter)
            writer.add_scalar('Reward/Validation Episode Reward - easy', epi_r1, n_iter)
            writer.add_scalar('Reward/Validation Episode Reward - subopt', epi_r2, n_iter)
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




