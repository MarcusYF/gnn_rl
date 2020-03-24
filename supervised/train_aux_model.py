import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from k_cut import *
import time
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm
from Analysis.episode_stats import test_summary
from DQN import to_cuda
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name = 'dqn_3by3_0229_base1'
save_folder = 'aux_test16'
model_version = str(30000)
k = 3
m = 3
ajr = 8
h = 32
mode = 'complete'
q_net = 'mlp'
batch_size = 200
trial_num = 1
sample_episode = batch_size * trial_num
gnn_step = 3
episode_len = 10
explore_prob = 0.0
Temperature = 0.0000005
n_epoch = 2000
save_ckpt_step = 1000
lr = 0.01

folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + model_name + '/'
with open(folder + 'dqn_' + model_version, 'rb') as model_file:
    alg = pickle.load(model_file)

# test summary
problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, mode=mode, sample_episode=sample_episode)
test = test_summary(alg=alg, problem=problem, q_net=q_net, forbid_revisit=0)

model = DQNet(k=k, m=m, ajr=ajr, num_head=4, hidden_dim=h).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

absroot = os.path.dirname(os.getcwd())
path = absroot + '/Models/test/'
if not os.path.exists(path):
    os.makedirs(path)

Loss = []
t = 0

writer = SummaryWriter(os.path.abspath(os.path.join(os.getcwd(), "..")) + '/runs/' + save_folder)

data_path = '/p/reinforcement/data/gnn_rl/aux_3by3_0229_base1/'
# for i in tqdm(range(100)):
#     x = []
#     y = []
#     for j in tqdm(range(10)):
#         bg = to_cuda(problem.gen_batch_graph(batch_size=batch_size))
#         test.run_test(problem=bg, trial_num=trial_num, batch_size=batch_size, gnn_step=gnn_step, episode_len=episode_len,
#                       explore_prob=explore_prob, Temperature=Temperature)
#         # test.show_result()
#
#         sway2_bg, s_eval_label = test.collect_sample_episode()
#         x.extend(sway2_bg)
#         y.extend(s_eval_label)
#     with open(path + 'batch_' + str(i), 'wb') as data_file:
#         pickle.dump([x, y], data_file)

def extract_manual_feature(g, k, m):
    batch_size = g.batch_size
    n = k * m
    nonzero_idx = [i for i in range(n ** 2) if i % (n + 1) != 0]
    # centroid = [g.ndata['h'][g.ndata['label'][:, ki] == 1].view(batch_size, k, -1).sum(dim=1) / m for ki in
    #             range(k)]

    #  (b, k, n*(n-1))
    group_matrix_k = torch.bmm(
        g.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * k, n, 1),
        g.ndata['label'].view(batch_size, n, -1).transpose(1, 2).reshape(batch_size * k, n, 1).transpose(1,
                                                                                                                   2)).view(
        batch_size * k, -1)[:, nonzero_idx].view(batch_size, k, -1)
    #  (k, b*n*(n-1))
    remain_edges = group_matrix_k.transpose(0, 1).reshape(k, -1) * g.edata['d'][:, 0]
    #  (k, b)
    new_S_k = remain_edges.view(k, batch_size, -1).sum(dim=2).t()  # (b, k)
    new_maxd_k = remain_edges.view(k, batch_size, -1).max(dim=2).values.t()  # (b, k)
    tmp = remain_edges.view(k, batch_size, -1)
    new_mind_k = tmp.min(dim=2).values.t()  # (b, k)

    Sk = torch.cat([new_S_k, new_S_k.max(dim=1).values.unsqueeze(1),
    new_S_k.mean(dim=1).unsqueeze(1),
    new_S_k.std(dim=1).unsqueeze(1),
    new_S_k.min(dim=1).values.unsqueeze(1)], dim=1)  # (b, k+4)

    maxdk = torch.cat([new_maxd_k, new_maxd_k.max(dim=1).values.unsqueeze(1),
    new_maxd_k.mean(dim=1).unsqueeze(1),
    new_maxd_k.std(dim=1).unsqueeze(1),
    new_maxd_k.min(dim=1).values.unsqueeze(1)], dim=1)  # (b, k+4)

    mindk = torch.cat([new_mind_k, new_mind_k.max(dim=1).values.unsqueeze(1),
    new_mind_k.mean(dim=1).unsqueeze(1),
    new_mind_k.std(dim=1).unsqueeze(1),
    new_mind_k.min(dim=1).values.unsqueeze(1)], dim=1)  # (b, k+4)

    return torch.cat([Sk, maxdk, mindk], dim=1)  # (b, 3k+12)

for n_iter in tqdm(range(n_epoch)):

    if n_iter % save_ckpt_step == save_ckpt_step - 1:
        with open(path + save_folder + '_' + str(n_iter + 1), 'wb') as model_file:
            pickle.dump(model, model_file)
        t += 1

    T1 = time.time()

    if n_iter % 20 == 0:
        data_chunk_id = n_iter // 20  # 0 - 49
        with open(data_path + 'batch_' + str(data_chunk_id), 'rb') as data_file:
            d = pickle.load(data_file)
            b_s = len(d[1]) // 20

    batch_id = n_iter % 20
    sway2_bg, s_eval_label = d[0][batch_id*b_s:(batch_id+1)*b_s], d[1][batch_id*b_s:(batch_id+1)*b_s]

    bg = dgl.batch(sway2_bg)


    dummy_actions = torch.zeros(bg.batch_size, 2).int().cuda()
    S_enc, _, _, _ = alg.forward(bg, dummy_actions, gnn_step=3, aux_output=True)

    actions = problem.get_legal_actions(state=bg)
    _, _, _, q = alg.forward(bg, actions, gnn_step=3)

    # select top-k largest q-values
    q_topk = q.view(bg.batch_size, -1).sort(dim=1).values[:, -5:]  # (b, 5)
    # q_bottomk = q.view(bg.batch_size, -1).sort(dim=1).values[:, :5]  # (b, 5)
    q_a0 = q.view(bg.batch_size, -1)[:, -1:]  # (b, 1)
    q_avg = q.view(bg.batch_size, -1).mean(dim=1).unsqueeze(1)  # (b, 1)
    q_std = q.view(bg.batch_size, -1).std(dim=1).unsqueeze(1)  # (b, 1)
    # manual_features = extract_manual_feature(bg, k, m)  # (b, 3k+12)

    S_enc = torch.cat([S_enc, q_a0, q_topk, q_avg, q_std], dim=1)
    _, _, _, y = model.forward_state_eval(bg, S_enc, gnn_step=3)

    T2 = time.time()

    L = torch.pow(y - torch.tensor(s_eval_label).cuda(), 2)
    L = L.sum()
    optimizer.zero_grad()
    L.backward()
    Loss.append(L.detach().item())
    model.h_residual.append(Loss[-1])
    optimizer.step()
    T3 = time.time()

    writer.add_scalar('Loss/Q-Loss', Loss[-1], n_iter)
    writer.add_scalar('Time/Running Time per Epoch', T3 - T1, n_iter)
    print('\nEpoch: {}. Loss: {}. T: {}.'
              .format(n_iter
               , np.round(Loss[-1], 2)
               , np.round(T2-T1, 3)
                , np.round(T3-T2, 3)))

