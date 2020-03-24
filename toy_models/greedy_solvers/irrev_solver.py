import numpy as np
import torch
from tqdm import tqdm
from toy_models.Qiter import state2QtableKey
from k_cut import *

k = 10
m = 10
n = k * m
ajr = 8
h = 32
mode = 'complete'
q_net = 'mlp'
batch_size = 100
trial_num = 1
sample_episode = batch_size * trial_num
gnn_step = 3
episode_len = 10
explore_prob = 0.0
Temperature = 0.0000005

def mca_irrev_kcut(k, m, x, dist_mat=None, print_info=True):

    n = k * m

    uk = x

    group = []
    select_hash = []
    select_idx = []
    for i in range(k):

        sqr_dist = F.relu(-2*torch.mm(uk, uk.t()) + (uk ** 2).sum(axis=1).reshape(n,1) + (uk ** 2).sum(axis=1).reshape(1,n))
        nearest_k_indice = sqr_dist.sort(dim=1).indices[:, :m].sort(dim=1).values
        sorted_nearest_k_indice = nearest_k_indice[nearest_k_indice[:, 0].sort().indices, :]

        # hashed id for each partition
        hashed_partitions = torch.sum(sorted_nearest_k_indice * torch.tensor([n**(i-1) for i in range(m, 0, -1)]), dim=1)
        hashed_partitions = torch.tensor([x % 2147483648 for x in hashed_partitions])
        b = torch.bincount(hashed_partitions)
        if i > 0:
            b[torch.tensor(select_hash)] = 0
        c = b.argmax()
        select_idx.append((hashed_partitions == c).int().argmax())
        select_hash.append(c)
        group_id = sorted_nearest_k_indice[select_idx[-1], :]
        group.append(group_id.unsqueeze(0))
        uk[group_id] += 10 ** (i+3)

    if print_info:
        print(group)

    if len(set(torch.cat(group, dim=0).flatten())) == n:
        a = np.zeros(n)
        for j in range(k):
            a[group[j]] = j
        return [int(x) for x in a]
    else:
        print('partition failed!')
        return None


# bingo = 0
# for i in tqdm(range(len(validation_problem1))):
#
#     k, m = 3, 3
#     x = validation_problem0[i][0].g.ndata['x']
#     result = sdp_kcut(k=3, m=3, x=x, print_info=False)
#
#     if state2QtableKey(result) == state2QtableKey(validation_problem0[i][1]):
#         bingo += 1

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h, mode=mode, sample_episode=sample_episode)
test_bg = problem.gen_batch_graph(batch_size=batch_size)
init_S = problem.calc_batchS(bg=test_bg)
print(sum(init_S))

sdp_label = []
for i in tqdm(range(batch_size)):
    result = mca_irrev_kcut(k=k, m=m, x=test_bg.ndata['x'][n*i:n*(i+1),:], print_info=False)
    sdp_label.extend(result)
tmp_bg = dc(test_bg)
problem.set_batch_label(tmp_bg, torch.tensor(sdp_label))
end_S = problem.calc_batchS(bg=tmp_bg)
print(sum(end_S))

print((sum(init_S)-sum(end_S))/sum(init_S))