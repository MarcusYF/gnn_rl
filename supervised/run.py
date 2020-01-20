import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import pickle
import time
from k_cut import *
import dgl
import torch
from tqdm import tqdm
from supervised.parse_sample import data_handler, map_psar2g
import argparse


parser = argparse.ArgumentParser(description="GNN with Supervised Learning")
parser.add_argument('--save_folder', default='test')
parser.add_argument('--gpu', default='0', help="")
parser.add_argument('--action_type', default='swap', help="")
parser.add_argument('--k', default=3, help="size of K-cut")
parser.add_argument('--m', default=3, help="cluster size")
parser.add_argument('--ajr', default=5, help="")
parser.add_argument('--h', default=16, help="hidden dimension")
parser.add_argument('--extend_h', default=True)
parser.add_argument('--use_x', default=True)
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--n_epoch', default=20000)
parser.add_argument('--save_ckpt_step', default=5000)
parser.add_argument('--num_chunks', default=30, help='')
parser.add_argument('--batch_size', default=1000, help='')
parser.add_argument('--gnn_step', default=3, help='')

args = vars(parser.parse_args())

save_folder = args['save_folder']
gpu = args['gpu']
action_type = args['action_type']
k = int(args['k'])
m = int(args['m'])
ajr = int(args['ajr'])
h = int(args['h'])
extend_h = bool(args['extend_h'])
use_x = bool(args['use_x'])
lr = args['lr']    # learning rate
n_epoch = int(args['n_epoch'])
save_ckpt_step = int(args['save_ckpt_step'])
num_chunks = int(args['num_chunks'])
batch_size = int(args['batch_size'])
gnn_step = int(args['gnn_step'])

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
parent_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

# Construct data reader
dh = data_handler(path=parent_path + '/Data/'
                  , num_chunks=num_chunks
                  , batch_size=batch_size)
dh.build_one_pass_index()

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h)
model = DQNet(k=k, m=m, ajr=ajr, num_head=4, hidden_dim=h, extended_h=extend_h, use_x=use_x).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
path = parent_path + '/Models/' + save_folder + '/'
if not os.path.exists(path):
    os.makedirs(path)
with open(path + 'sup_0', 'wb') as model_file:
    pickle.dump(model, model_file)


def run_supervised_training(model):
    Loss = []
    for i in tqdm(range(n_epoch)):

        if i % save_ckpt_step == save_ckpt_step - 1:
            with open(path + 'sup_' + str(i + 1), 'wb') as model_file:
                pickle.dump(model, model_file)
            with open(path + 'sup_' + str(i + 1), 'rb') as model_file:
                model = pickle.load(model_file)


        T1 = time.time()
        current_batch = dh.sample_batch(batch_idx=i)

        batch_state = dgl.batch([map_psar2g(pasr) for pasr in current_batch])
        batch_action = [torch.tensor(pasr.a).unsqueeze(0) for pasr in current_batch]
        batch_action = torch.cat(batch_action, axis=0).cuda()
        target_Q = torch.tensor([pasr.r for pasr in current_batch]).cuda()


        S_a_encoding, h1, h2, Q_sa = model(batch_state, batch_action)
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


if __name__ == '__main__':
    run_supervised_training(model)


