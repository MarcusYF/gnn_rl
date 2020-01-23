import os
import pickle
import time
from k_cut import *
import dgl
import torch
from tqdm import tqdm
from supervised.parse_sample import data_handler, map_psar2g
from torch.optim import lr_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
action_type = 'swap'
loss_fc = 'pairwise'
k = 3
m = 3
ajr = 5
hidden_dim = 16
extended_h = True
use_x = False
lr = 1e-4
n_epoch = 20000
save_ckpt_step = 5000
num_chunks = 1
batch_size = 1000
bundle_size = 500
gnn_step = 3

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
model = DQNet(k=k, m=m, ajr=ajr, num_head=4, hidden_dim=hidden_dim, extended_h=extended_h, use_x=use_x).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Construct data reader
# dh = data_handler(num_chunks=num_chunks, batch_size=batch_size)
# dh.build_one_pass_index()


absroot = os.path.dirname(os.getcwd())
path = absroot + '/Models/test/'
if not os.path.exists(path):
    os.makedirs(path)
with open(path + 'sup_0', 'wb') as model_file:
    pickle.dump(model, model_file)

Loss = []

for i in tqdm(range(n_epoch)):

    inner_i = i % bundle_size
    outer_i = i // bundle_size

    if inner_i == 0:
        with open('/net/bigtemp/fy4bc/Data/gnn_rl/sup_B1000/batch_' + str(outer_i), 'rb') as m:
            data_bundle = pickle.load(m)

    if i % save_ckpt_step == save_ckpt_step - 1:
        with open(path + 'sup_' + str(i + 1), 'wb') as model_file:
            pickle.dump(model, model_file)
        with open(path + 'sup_' + str(i + 1), 'rb') as model_file:
            model = pickle.load(model_file)


    T1 = time.time()

    # current_batch = dh.sample_batch(batch_idx=i)
    #
    # batch_state = dgl.batch([map_psar2g(psaq) for psaq in current_batch])
    #
    # batch_action = [torch.tensor(psaq.a).unsqueeze(0) for psaq in current_batch]
    # batch_action = torch.cat(batch_action, axis=0).cuda()
    # if loss_fc == 'pairwise':
    #     batch_best_action = [torch.tensor(psaq.best_a).unsqueeze(0) for psaq in current_batch]
    #     batch_best_action = torch.cat(batch_best_action, axis=0).cuda()
    #     batch_action = torch.cat([batch_action, batch_best_action], axis=1).view(-1, 2).cuda()

    batch_state = data_bundle[inner_i][0]
    batch_action = data_bundle[inner_i][1].cuda()
    # batch_best_action = data_bundle[inner_i][2].cuda()
    target_Q = data_bundle[inner_i][3].cuda()
    best_Q = data_bundle[inner_i][4].cuda()

    S_a_encoding, h1, h2, Q_sa = model(batch_state, batch_action)

    if loss_fc == 'pairwise':
        target_Q = Q_sa[0::2]
        best_Q = Q_sa[1::2]
        L = F.relu(target_Q - best_Q)
    else:
        # target_Q = torch.tensor([psaq.q for psaq in current_batch]).cuda()
        # assign different weight to different target q-values
        # best_Q = torch.tensor([psaq.best_q for psaq in current_batch]).cuda()
        L = torch.pow(Q_sa - target_Q, 2) / (0.1 + best_Q - target_Q) \
         + 1.6 * F.relu(Q_sa - best_Q)

    L = L.sum()
    optimizer.zero_grad()
    L.backward()
    Loss.append(L.detach().item())
    model.h_residual.append(Loss[-1])
    optimizer.step()
    T6 = time.time()

    print('\nEpoch: {}. Loss: {}. T: {}.'
              .format(i
               , np.round(Loss[-1], 2)
               , np.round(T6-T1, 3)))

