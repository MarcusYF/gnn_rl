import os
import pickle
import time
from k_cut import *
import torch
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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


problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
model = DQNet(k=k, m=m, ajr=ajr, num_head=4, hidden_dim=hidden_dim, extended_h=extended_h, use_x=use_x).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

with open('/p/reinforcement/data/gnn_rl/sup_split_100graphs/batch_0', 'rb') as m:
    data_bundle = pickle.load(m)


Loss = []
g_i = 0
for i in tqdm(range(n_epoch)):


    T1 = time.time()

    batch_state = data_bundle[g_i][0]
    batch_action = data_bundle[g_i][1].cuda()
    # batch_best_action = data_bundle[inner_i][2].cuda()
    target_Q = data_bundle[g_i][3].cuda()
    best_Q = data_bundle[g_i][4].cuda()

    S_a_encoding, h1, h2, Q_sa = model.forward_vanilla(batch_state, batch_action)

    if loss_fc == 'pairwise':
        target_Q = Q_sa[0::2]
        best_Q = Q_sa[1::2]
        L = F.relu(target_Q - best_Q) # 2
    else:
        L = torch.pow(Q_sa - target_Q, 2) / (0.01 + best_Q - target_Q) \
         + 20 * F.relu(Q_sa - best_Q) # 3
        # L = torch.pow(Q_sa - target_Q, 2) / Q_sa.shape[0] 4

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

# with open('/p/reinforcement/data/gnn_rl/model/sup' + 'model_2', 'wb') as model_file:
#     pickle.dump(model, model_file)
problem = KCut_DGL(k=k, m=3, adjacent_reserve=ajr, hidden_dim=hidden_dim)

g = generate_G(k=3, m=3, adjacent_reserve=5, hidden_dim=16, random_sample_node=False, x=data_bundle[0][0].ndata['x'][:9, :], random_init_label=True)
problem.g = g
problem.reset_label(label=[0,0,0,1,1,1,2,2,2])

data = data_bundle[g_i]

 state2QtableKey(data[0].ndata['label'][9:18,:].argmax(dim=1).cpu().numpy())

test = test_summary(alg=model, problem=[problem], num_instance=1)
test.run_test(episode_len=50, explore_prob=.0, time_aware=False, softmax_constant=1e10)
test.show_result()

batch_state

_,_,_,Q_sa = model.forward_vanilla(dgl.batch([to_cuda(problem.g)]), problem.get_legal_actions().cuda())



