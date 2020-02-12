import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import pickle
import time
import json
from k_cut import *
import dgl
import torch
from tqdm import tqdm
from supervised.parse_sample import data_handler, map_psar2g
import argparse
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# python run.py --save_folder=sup_0123_vanilla --gpu=0 --q_net=mlp

parser = argparse.ArgumentParser(description="GNN with Supervised Learning")
parser.add_argument('--save_folder', default='test')
parser.add_argument('--gpu', default='0', help="")
parser.add_argument('--action_type', default='swap', help="")
parser.add_argument('--k', default=3, help="size of K-cut")
parser.add_argument('--m', default=3, help="cluster size")
parser.add_argument('--ajr', default=5, help="")
parser.add_argument('--h', default=16, help="hidden dimension")
parser.add_argument('--extend_h', default=True)
parser.add_argument('--use_x', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--lr_half_time', default=0)
parser.add_argument('--n_epoch', default=20000)
parser.add_argument('--save_ckpt_step', default=5000)
parser.add_argument('--num_chunks', default=100, help='')
parser.add_argument('--batch_size', default=1000, help='')
parser.add_argument('--gnn_step', default=3, help='')

parser.add_argument('--q_net', default='mlp', help="")
parser.add_argument('--loss_fc', default='-pairwise', help="")
parser.add_argument('--alpha', type=float, default=0.01, help='')
parser.add_argument('--lambda', type=float, default=20, help='')

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
lr = args['lr']
lr_half_time = int(args['lr_half_time'])
n_epoch = int(args['n_epoch'])
save_ckpt_step = int(args['save_ckpt_step'])
num_chunks = int(args['num_chunks'])
batch_size = int(args['batch_size'])
gnn_step = int(args['gnn_step'])

q_net = args['q_net']
loss_fc = args['loss_fc']
alpha = float(args['alpha'])
lambd = float(args['lambda'])

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
parent_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))


# problem
# problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=h)

# Initialize model and optimizer
model = DQNet(k=k, m=m, ajr=ajr, num_head=4, hidden_dim=h, extended_h=extend_h, use_x=use_x).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if lr_half_time == 0:
    gamma = 1
else:
    gamma = 0.5**(1 / lr_half_time)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


# Initialize save path
path = parent_path + '/Models/' + save_folder + '/'
if not os.path.exists(path):
    os.makedirs(path)
with open(path + 'sup_0', 'wb') as model_file:
    pickle.dump(model, model_file)

# record running hyper-parameters
with open(path + 'params', 'w') as params_file:
    params_file.write(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime()))
    params_file.write('\n------------------------------------\n')
    params_file.write(json.dumps(args))
    params_file.write('\n------------------------------------\n')

def run_supervised_training(model):
    # Construct data reader
    dh = data_handler(path=parent_path + '/Data/'
                      , num_chunks=num_chunks
                      , batch_size=batch_size)
    dh.build_one_pass_index()

    Loss = []
    for i in tqdm(range(n_epoch)):

        if i % save_ckpt_step == save_ckpt_step - 1:
            with open(path + 'sup_' + str(i + 1), 'wb') as model_file:
                pickle.dump(model, model_file)
            with open(path + 'sup_' + str(i + 1), 'rb') as model_file:
                model = pickle.load(model_file)

        T1 = time.time()
        current_batch = dh.sample_batch(batch_idx=i)

        batch_state = dgl.batch([map_psar2g(psaq) for psaq in current_batch])

        batch_action = [torch.tensor(psaq.a).unsqueeze(0) for psaq in current_batch]
        batch_action = torch.cat(batch_action, axis=0).cuda()
        if loss_fc == 'pairwise':
            batch_best_action = [torch.tensor(psaq.best_a).unsqueeze(0) for psaq in current_batch]
            batch_best_action = torch.cat(batch_best_action, axis=0).cuda()
            batch_action = torch.cat([batch_action, batch_best_action], axis=1).view(-1, 2).cuda()

        if q_net == 'mlp':
            S_a_encoding, h1, h2, Q_sa = model.forward_vanilla(batch_state, batch_action)
        else:
            S_a_encoding, h1, h2, Q_sa = model.forward(batch_state, batch_action)
        target_Q = torch.tensor([psaq.q for psaq in current_batch]).cuda()
        # assign different weight to different target q-values
        best_Q = torch.tensor([psaq.best_q for psaq in current_batch]).cuda()

        if loss_fc == 'pairwise':
            cur_a_Q = Q_sa[0::2]
            best_a_Q = Q_sa[1::2]
            L = torch.pow(cur_a_Q - target_Q, 2) \
                + torch.pow(best_a_Q - best_Q, 2) * alpha \
                + lambd * F.relu(cur_a_Q - best_a_Q)
        else:
            L = torch.pow(Q_sa - target_Q, 2) / (alpha + best_Q - target_Q) \
                + lambd * F.relu(Q_sa - best_Q)

        L = L.sum()

        optimizer.zero_grad()
        L.backward()
        Loss.append(L.detach().item())
        model.h_residual.append(Loss[-1])
        optimizer.step()
        scheduler.step()
        T6 = time.time()

        print('\nEpoch: {}. Loss: {}. T: {}.'
                  .format(i
                   , np.round(Loss[-1], 2)
                   , np.round(T6-T1, 3)))


def run_fast_supervised_training(model):
    Loss = []
    bundle_size = 500
    for i in tqdm(range(n_epoch)):

        inner_i = i % bundle_size
        outer_i = i // bundle_size

        if inner_i == 0:
            with open('/net/bigtemp/fy4bc/Data/gnn_rl/sup_B1000/batch_' + str(outer_i), 'rb') as m:
                data_bundle = pickle.load(m)

        if i % save_ckpt_step == save_ckpt_step - 1 or (outer_i==38 and inner_i==bundle_size-1):
            with open(path + 'sup_' + str(i + 1), 'wb') as model_file:
                pickle.dump(model, model_file)
            with open(path + 'sup_' + str(i + 1), 'rb') as model_file:
                model = pickle.load(model_file)

        T1 = time.time()

        batch_state = data_bundle[inner_i][0]
        batch_action = data_bundle[inner_i][1].cuda()
        target_Q = data_bundle[inner_i][3].cuda()
        best_Q = data_bundle[inner_i][4].cuda()

        if loss_fc == 'pairwise':
            batch_best_action = data_bundle[inner_i][2].cuda()
            batch_action = torch.cat([batch_action, batch_best_action], axis=1).view(-1, 2).cuda()

        S_a_encoding, h1, h2, Q_sa = model(batch_state, batch_action)

        if loss_fc == 'pairwise':
            cur_a_Q = Q_sa[0::2]
            best_a_Q = Q_sa[1::2]
            L = torch.pow(cur_a_Q - target_Q, 2) \
                + torch.pow(best_a_Q - best_Q, 2) * alpha \
                + lambd * F.relu(cur_a_Q - best_a_Q)
        else:
            L = torch.pow(Q_sa - target_Q, 2) / (alpha + best_Q - target_Q) \
                + lambd * F.relu(Q_sa - best_Q)

        L = L.sum()

        optimizer.zero_grad()
        L.backward()
        Loss.append(L.detach().item())
        model.h_residual.append(Loss[-1])
        optimizer.step()
        scheduler.step()
        T6 = time.time()

        print('\nEpoch: {}. Loss: {}. T: {}.'
                  .format(i
                   , np.round(Loss[-1], 2)
                   , np.round(T6-T1, 3)))

if __name__ == '__main__':
    run_supervised_training(model)
    # run_fast_supervised_training(model)


