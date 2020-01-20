import os
import pickle
from k_cut import *
from DQN import DQN, to_cuda
import dgl
import torch



path = os.path.dirname(os.getcwd()) + '/Data/'

data_paths = []
for root, dirs, filename in os.walk(path):
    if filename:
        data_paths.extend([root + '/' + f for f in filename])


for data_path in data_paths[0:1]:
    with open(data_path, 'rb') as data_file:
        data_chunk = pickle.load(data_file)



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
action_type = 'swap'
k = 3
m = 3
ajr = 5
hidden_dim = 16
extended_h = True
time_aware = False
a = 1
gamma = 0.90
lr = 1e-4
replay_buffer_max_size = 100 #
n_epoch = 5000
save_ckpt_step = 500
eps = np.linspace(1, 0.1, 2500) #
target_update_step = 5
batch_size = 1000
grad_accum = 1
sample_batch_episode = False
num_episodes = 10
episode_len = 50
gnn_step = 3
q_step = 1
ddqn = False

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
model = DQNet(k=k, m=m, ajr=ajr, num_head=4, hidden_dim=hidden_dim, extended_h=extended_h).cuda()

batch_g = [to_cuda(data_chunk[i][0].g) for i in range(len(data_chunk))]
for g in batch_g:
    g.ndata['h'] = torch.zeros((g.number_of_nodes(), hidden_dim)).cuda()

batch_state = dgl.batch(batch_g)
batch_action = [problem.get_legal_actions(state=g) for g in batch_g]
batch_action = torch.cat(batch_action, axis=0).cuda()

S_a_encoding, h1, h2, Q_sa = model(batch_state, batch_action)


optm = torch.optim.Adam(model.parameters(), lr=lr)
for i in 100:
    optm.zero_grad()
    L = torch.pow(R + Q, 2).sum()
    L.backward()
    optm.step()


