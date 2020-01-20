import os
import pickle
from k_cut import *
from DQN import to_cuda
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
            q_tb = self.data_chunks[pi][1]
            s = list(q_tb.keys())[si]
            a = list(q_tb[s].keys())[ai]
            r = q_tb[s][a][1]
            if self.rotate_label:
                m = self.label_map[label_permute_seed[i]]
                s = [m[x] for x in QtableKey2state(s)]
            else:
                s = QtableKey2state(s)
            batch_sample.append(Psar(self.data_chunks[pi][0], s, a, r))
        return batch_sample


def map_psar2g(psar, hidden_dim=16, rewire_edges=False):

    # TODO: if use cluster topology in gnn,
    #  then rewire_edges=True is required(would be time consuming)
    psar.p.reset_label(label=psar.s, calc_S=False, rewire_edges=rewire_edges)
    g = to_cuda(psar.p.g)
    g.ndata['h'] = torch.zeros((g.number_of_nodes(), hidden_dim)).cuda()
    return g
