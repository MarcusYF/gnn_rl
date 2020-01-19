import os
import pickle
import time
from k_cut import *
from DQN import DQN, to_cuda
import dgl
import torch
from tqdm import tqdm
import random
from Qiter_swap import QtableKey2state
from dataclasses import dataclass

@dataclass
class Psar:
    p: KCut_DGL
    s: list
    a: tuple
    r: float


class data_handler():

    def __init__(self, path=None, data_chunks=None, num_chunks=100, batch_size=100, rotate_label=True):

        if data_chunks is None:
            # read data chunks from given path
            if path is None:
                path = os.path.dirname(os.getcwd()) + '/Data/'

            data_paths = []
            for root, dirs, filename in os.walk(path):
                if filename:
                    data_paths.extend([root + '/' + f for f in filename])

            self.data_chunks = []  # [(graph, Q-table, error), ...]
            num_chunks = min(num_chunks, len(data_paths))
            for data_path in tqdm(data_paths[0:num_chunks]):
                with open(data_path, 'rb') as data_file:
                    try:
                        self.data_chunks.extend(pickle.load(data_file))
                    except:
                        print('Deprecated data block detected: ', data_path)
        else:
            self.data_chunks = data_chunks

        self.num_instance = self.data_chunks.__len__()
        self.num_state = self.data_chunks[-1][1].keys().__len__()
        self.num_action = self.data_chunks[-1][1].get('0,0,0,1,1,1,2,2,2').keys().__len__()
        self.n = self.num_instance * self.num_state * self.num_action
        self.Q_table_size = self.num_state * self.num_action
        self.one_pass_idx = list(range(self.n))
        self.data_pass = 0
        self.rotate_label = rotate_label
        self.label_map = [{0:0, 1:1, 2:2}
             , {0:0, 1:2, 2:1}
             , {0:1, 1:0, 2:2}
             , {0:1, 1:2, 2:0}
             , {0:2, 1:0, 2:1}
             , {0:2, 1:1, 2:0}]
        self.batch_size = batch_size
        self.label_permute_seed = np.random.randint(0, 6, self.batch_size)


    def build_one_pass_index(self):
        print('Building one pass index...')
        random.shuffle(self.one_pass_idx)
        print('Indexing complete.')

    def sample_batch_index(self, batch_idx=0):
        batch_size = self.batch_size
        off_set = batch_idx * batch_size
        if batch_size + off_set > self.n:
            self.data_pass += 1
            print('One pass over. Re-indexing...')
            self.build_one_pass_index()
        off_set = batch_idx * batch_size - self.n * self.data_pass
        return self.one_pass_idx[off_set: off_set + batch_size]

    def sample_batch(self, batch_idx=0):
        label_permute_seed = np.random.randint(0, 6, self.batch_size)
        batch_size = self.batch_size
        sample_index = self.sample_batch_index(batch_idx=batch_idx)
        p_s_a_idx = [(s // (self.Q_table_size)
          , s % (self.Q_table_size) // self.num_action
          , s % (self.Q_table_size) % self.num_action) for s in sample_index]

        batch_sample = []
        for i in range(batch_size):
            pi, si, ai = p_s_a_idx[i]
            p = self.data_chunks[pi][0]
            q_tb = self.data_chunks[pi][1]
            s = list(q_tb.keys())[si]
            a = list(q_tb[s].keys())[ai]
            r = q_tb[s][a][1]
            if self.rotate_label:
                m = self.label_map[label_permute_seed[i]]
                s = [m[x] for x in QtableKey2state(s)]
            else:
                s = QtableKey2state(s)
            batch_sample.append(Psar(p, s, a, r))
        return batch_sample


def map_psar2g(psar, hidden_dim=16, rewire_edges=False):

    # TODO: if use cluster topology in gnn,
    #  then rewire_edges=True is required(would be time consuming)
    psar.p.reset_label(label=psar.s, calc_S=False, rewire_edges=rewire_edges)
    g = to_cuda(psar.p.g)
    g.ndata['h'] = torch.zeros((g.number_of_nodes(), hidden_dim)).cuda()
    return g



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
action_type = 'swap'
k = 3
m = 3
ajr = 5
hidden_dim = 16
extended_h = True
lr = 1e-4
n_epoch = 5000
save_ckpt_step = 500
batch_size = 1000
gnn_step = 3

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
model = DQNet(k=k, m=m, ajr=ajr, num_head=4, hidden_dim=hidden_dim, extended_h=extended_h).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Construct data reader
dh = data_handler(data_chunks=data_chunks, batch_size=batch_size)
dh.build_one_pass_index()

absroot = os.path.dirname(os.getcwd())
path = absroot + '/Models/sup_base/'
if not os.path.exists(path):
    os.makedirs(path)
with open(path + 'sup_0', 'wb') as model_file:
    pickle.dump(model, model_file)

for i in tqdm(range(n_epoch)):

    if i % save_ckpt_step == save_ckpt_step - 1:
        with open(path + 'sup_' + str(i + 1), 'wb') as model_file:
            pickle.dump(alg, model_file)
        with open(path + 'sup_' + str(i + 1), 'rb') as model_file:
            alg = pickle.load(model_file)


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
    optimizer.step()
    T6 = time.time()

    print('\nEpoch: {}. Loss: {}. T: {}.'
              .format(i
               , np.round(L.detach().item(), 2)
               # , np.round(T2-T1, 3)
               # , np.round(T3 - T2, 3)
               # , np.round(T4 - T3, 3), np.round(T5-T4, 3)
               , np.round(T6-T1, 3)))




