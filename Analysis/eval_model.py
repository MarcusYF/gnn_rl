from k_cut import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pickle
import torch
import os
from tqdm import tqdm
from Analysis.episode_stats import test_summary
from toy_models.ga_helpers.data_loader import dump_matrix
from toy_models.Qiter import vis_g
from DQN import to_cuda

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model_name = 'dqn_5by6_0217_weightS'
model_name = 'dqn_3by3_0218_t9'
model_name = 'dqn_3by3_0223_pauseA_w'

model_name = 'dqn_3by3_0223_pauseA_clipQ'
model_name = 'dqn_3by3_0223_base_in'
model_name = 'dqn_3by3_0229_base1'
# model_name = 'dqn_3by3_0225_h32'
model_version = str(30000)
k = 3
m = 3
ajr = 8
h = 32
mode = 'complete'
q_net = 'mlp'
batch_size = 100
trial_num = 1
sample_episode = batch_size * trial_num
gnn_step = 3
episode_len = 50
explore_prob = 0.0
Temperature = 0.0000005

folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + model_name + '/'
with open(folder + 'dqn_' + model_version, 'rb') as model_file:
    alg = pickle.load(model_file)

# test summary
problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, mode=mode, sample_episode=sample_episode)
test = test_summary(alg=alg, problem=problem, q_net=q_net, forbid_revisit=0)

with open('/p/reinforcement/data/gnn_rl/model/test_data/3by3/0', 'rb') as valid_file:
    validation_problem0 = pickle.load(valid_file)
with open('/p/reinforcement/data/gnn_rl/model/test_data/3by3/1', 'rb') as valid_file:
    validation_problem1 = pickle.load(valid_file)
bg_hard = to_cuda(dgl.batch([p[0].g for p in validation_problem0[:batch_size]]))
bg_easy = to_cuda(dgl.batch([p[0].g for p in validation_problem1[:batch_size]]))
bg_subopt = []
bg_opt = []
for i in range(batch_size):
    gi = to_cuda(validation_problem0[:batch_size][i][0].g)
    problem.reset_label(g=gi, label=validation_problem0[:batch_size][i][2])
    bg_subopt.append(gi)
    gj = to_cuda(validation_problem0[:batch_size][i][0].g)
    problem.reset_label(g=gj, label=validation_problem0[:batch_size][i][1])
    bg_opt.append(gj)
bg_subopt = dgl.batch(bg_subopt)
bg_opt = dgl.batch(bg_opt)

if ajr == 8:
    bg_hard.edata['e_type'][:, 0] = torch.ones(k * m * ajr * bg_hard.batch_size)
    bg_easy.edata['e_type'][:, 0] = torch.ones(k * m * ajr * bg_easy.batch_size)
    bg_subopt.edata['e_type'][:, 0] = torch.ones(k * m * ajr * bg_subopt.batch_size)
    bg_opt.edata['e_type'][:, 0] = torch.ones(k * m * ajr * bg_opt.batch_size)


# random validation set
bg = to_cuda(problem.gen_batch_graph(batch_size=batch_size))
test.run_test(problem=bg, trial_num=trial_num, batch_size=batch_size, gnn_step=gnn_step, episode_len=episode_len, explore_prob=explore_prob, Temperature=Temperature)
test.show_result()

# easy validation set
test.run_test(problem=to_cuda(bg_easy), trial_num=1, batch_size=100, gnn_step=gnn_step,
              episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
epi_r1 = test.show_result()
best_hit1 = test.compare_opt(validation_problem1)

# hard validation set
test.run_test(problem=to_cuda(bg_hard), trial_num=1, batch_size=100, gnn_step=gnn_step,
              episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
epi_r0 = test.show_result()
best_hit0 = test.compare_opt(validation_problem0)

# subopt validation set

test = test_summary(alg=alg, problem=problem, q_net=q_net, forbid_revisit=0)
test.run_test(problem=to_cuda(bg_subopt), trial_num=1, batch_size=100, gnn_step=gnn_step,
              episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
epi_r2 = test.show_result()
best_hit2 = test.compare_opt(validation_problem0)

subopt_Vs = [torch.mean(test.episodes[i].q_pred[0]).cpu().item() for i in range(100)]

# opt validation set
test1 = test_summary(alg=alg, problem=problem, q_net=q_net, forbid_revisit=0)
test1.run_test(problem=to_cuda(bg_opt), trial_num=1, batch_size=100, gnn_step=gnn_step,
              episode_len=episode_len, explore_prob=0.0, Temperature=1e-8)
epi_r3 = test1.show_result()
best_hit3 = test1.compare_opt(validation_problem0)
opt_Vs = [torch.mean(test1.episodes[i].q_pred[0]).cpu().item() for i in range(100)]

for i in range(100):
    print(subopt_Vs[i] - opt_Vs[i])

a = torch.zeros(100, 10)
win = []
sway = []
zero = []
fail = []
for i in range(100):
    gain_ep = np.cumsum(test.episodes[i].reward_seq)
    if max(gain_ep) > 0:
        win.append(i)
    elif max(gain_ep) == 0 and min(gain_ep) < 0:
        sway.append(i)
    elif max(gain_ep) == 0 and min(gain_ep) == 0:
        zero.append(i)
    else:
        fail.append(i)
        # print(test.episodes[i].reward_seq[:10])
    a[i, :10] = torch.tensor(test.episodes[i].reward_seq[:10])

opt_q_mean = []
opt_q_std = []
opt_q_pause = []
subopt_q_mean = []
subopt_q_std = []
subopt_q_pause = []

subopt_q_mean_win = []
subopt_q_mean_sway = []
subopt_q_mean_zero = []
subopt_q_mean_fail = []
for i in range(100):
    opt_q_mean.append(test1.episodes[i].q_pred[0].mean())
    opt_q_std.append(test1.episodes[i].q_pred[0].std())
    opt_q_pause.append(test1.episodes[i].q_pred[0][-1])
    subopt_q_mean.append(test.episodes[i].q_pred[0].mean())
    subopt_q_std.append(test.episodes[i].q_pred[0].std())
    subopt_q_pause.append(test.episodes[i].q_pred[0][-1])

    if i in win:
        subopt_q_mean_win.append(test.episodes[i].q_pred[0].mean())
    if i in sway:
        subopt_q_mean_sway.append(test.episodes[i].q_pred[0].mean())
    if i in zero:
        subopt_q_mean_zero.append(test.episodes[i].q_pred[0].mean())
    if i in fail:
        subopt_q_mean_fail.append(test.episodes[i].q_pred[0].mean())

torch.tensor(subopt_q_mean_win).mean()
torch.tensor(subopt_q_mean_sway).mean()
torch.tensor(subopt_q_mean_zero).mean()
torch.tensor(subopt_q_mean_fail).mean()
torch.tensor(subopt_q_mean).mean()
torch.tensor(opt_q_mean).mean()
# visualization
gi = 0
test.episodes[gi].action_seq
test.episodes[gi].reward_seq
test.cmpt_optimal(test.episodes[gi].init_state, '/Analysis/eval_model/opt')
test.test_greedy(test.episodes[gi].init_state, '/Analysis/eval_model/grd_')
test.test_dqn(alg, gi, 10, '/Analysis/eval_model/dqn_')
# visualization
path = os.path.abspath(os.path.join(os.getcwd())) + '/Analysis/eval_model/test1'
vis_g(problem, name=path, topo='cut')




g_i = 1
init_state = dc(test.episodes[g_i].init_state)
label_history = init_state.ndata['label'][test.episodes[g_i].label_perm]

problem.g = dc(init_state)
problem.reset_label(label=label_history[0].argmax(dim=1))
problem.calc_S()

actions = problem.get_legal_actions(state=problem.g)
S_a_encoding, h1, h2, Q_sa = alg.forward(to_cuda(problem.g), actions.cuda(), gnn_step=gnn_step)
print(Q_sa)
print(torch.argmax(Q_sa), actions[torch.argmax(Q_sa)])
problem.step(state=problem.g, action=actions[torch.argmax(Q_sa)])

problem.reset_label(label=label_history[50].argmax(dim=1))
problem.calc_S()

path = os.path.abspath(os.path.join(os.getcwd())) + '/Analysis/eval_model/test1'
vis_g(problem, name=path, topo='cut')

actions = problem.get_legal_actions(state=problem.g)
bg = dgl.batch([to_cuda(problem.g)] * actions.shape[0])
problem.sample_episode = actions.shape[0]
problem.gen_step_batch_mask()

problem.step_batch(states=bg, action=actions.cuda())
S_a_encoding, h1, h2, Q_sa = alg.forward(to_cuda(bg), actions.cuda(), gnn_step=gnn_step)

# run GA benchmark
problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, mode=mode, sample_episode=sample_episode)
_, _, sq_dist_matrix = dgl.transform.knn_graph(problem.g.ndata['x'], ajr + 1, extend_info=True)
mat_5by6 = (2 - torch.sqrt(F.relu(sq_dist_matrix, inplace=True))[0]).numpy().astype('float64')
m_path = os.path.abspath(os.path.join(os.getcwd())) + '/toy_models/ga_helpers/corr_mat/dqn_5by6.mat'
dump_matrix(mat_5by6, m_path)


for i in tqdm(range(500)):
    for j in range(50):
        if test.episodes[i].action_seq[j][0] + test.episodes[i].action_seq[j][1] == 0:
            print(i, j)


actions = problem.get_legal_actions(state=problem.g)
S_a_encoding, h1, h2, Q_sa = alg.forward(to_cuda(problem.g), actions.cuda(), gnn_step=gnn_step)
print(Q_sa)
print(torch.argmax(Q_sa), actions[torch.argmax(Q_sa)])
problem.step(state=problem.g, action=actions[torch.argmax(Q_sa)])