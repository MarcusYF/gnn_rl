import os
import pickle
import time
from k_cut import *
import dgl
import torch
from tqdm import tqdm
from supervised.parse_sample import data_handler, map_psar2g

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
action_type = 'swap'
k = 3
m = 3
ajr = 5
hidden_dim = 16
extended_h = True
lr = 1e-4
n_epoch = 20000
save_ckpt_step = 5000
num_chunks = 10
batch_size = 1000
gnn_step = 3

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
model = DQNet(k=k, m=m, ajr=ajr, num_head=4, hidden_dim=hidden_dim, extended_h=extended_h).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Construct data reader
dh = data_handler(num_chunks=num_chunks, batch_size=batch_size)
dh.build_one_pass_index()

absroot = os.path.dirname(os.getcwd())
path = absroot + '/Models/test/'
if not os.path.exists(path):
    os.makedirs(path)
with open(path + 'sup_0', 'wb') as model_file:
    pickle.dump(model, model_file)

Loss = []
for i in tqdm(range(n_epoch)):

    if i % save_ckpt_step == save_ckpt_step - 1:
        with open(path + 'sup_' + str(i + 1), 'wb') as model_file:
            pickle.dump(model, model_file)
        with open(path + 'sup_' + str(i + 1), 'rb') as model_file:
            model = pickle.load(model_file)


    T1 = time.time()
    current_batch = dh.sample_batch(batch_idx=i)

    T2 = time.time()
    batch_state = dgl.batch([map_psar2g(pasr) for pasr in current_batch])
    T3 = time.time()
    batch_action = [torch.tensor(pasr.a).unsqueeze(0) for pasr in current_batch]
    batch_action = torch.cat(batch_action, axis=0).cuda()
    target_Q = torch.tensor([pasr.r for pasr in current_batch]).cuda()
    T4 = time.time()


    S_a_encoding, h1, h2, Q_sa = model(batch_state, batch_action)
    T5 = time.time()
    optimizer.zero_grad()
    L = torch.pow(Q_sa - target_Q, 2).sum()
    L.backward()
    Loss.append(L.detach().item())
    model.h_residual.append(Loss[-1])
    optimizer.step()
    T6 = time.time()

    print('\nEpoch: {}. Loss: {}. T: {}.'
              .format(i
               , np.round(Loss[-1], 2)
               , np.round(T6-T1, 3)))
