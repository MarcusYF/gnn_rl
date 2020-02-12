import os
import pickle
import time
from k_cut import *
from DQN import *
import torch
from tqdm import tqdm
import numpy as np
from Analysis.episode_stats import test_summary
import random
import matplotlib.pyplot as plt
from Qiter_swap import state2QtableKey, QtableKey2state, gen_comb012


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
action_type = 'swap'
loss_fc = '-pairwise'
k = 3
m = 3
ajr = 5
hidden_dim = 16
extended_h = True
use_x = False
lr = 1e-4
n_epoch = 10000



problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
model = DQNet(k=k, m=m, ajr=ajr, num_head=4, hidden_dim=hidden_dim, extended_h=extended_h, use_x=use_x).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

with open('/p/reinforcement/data/gnn_rl/sup_split_100graphs/batch_0', 'rb') as m:
    data_bundle = pickle.load(m)


def find_state(label):
    # find the location for each state
    for i in range(280):
        tmp = data_bundle[g_i][0].ndata['label'][i*9*27:i*9*27+9].nonzero()[:, 1].cpu().numpy()
        if state2QtableKey(tmp) == state2QtableKey(label):
            return i

# compute state_s_value
num_state = 280 * 27
template = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])
all_state = set([state2QtableKey([int(i) for i in s[:-1].split(',')], reduce_rotate=True) for s in gen_comb012(3, 3, 3)])

g0 = dgl.unbatch(data_bundle[g_i][0])
problem.g = g0[0]  # g0[i] differs in initial states
state_s_value = {}.fromkeys(all_state)
for state in all_state:
    problem.reset_label(label=torch.tensor(QtableKey2state(state)).cuda())
    state_s_value[state] = problem.calc_S().item()

layer_state = sorted(state_s_value.items(), key=lambda x: x[1], reverse=True)

# compute state weights
state_list = []
state_weight = []
for i in range(280):
    _ = data_bundle[g_i][0].ndata['label'][i * 9 * 27:i * 9 * 27 + 9].nonzero()[:, 1].cpu().numpy()
    state_list.append(state2QtableKey(_))
    state_weight.append(1/(state_s_value[state_list[-1]]-2))  # [0.3, 1.5]
state_weight = torch.tensor(state_weight).unsqueeze(0).repeat(27, 1).t().flatten().cuda()




# train supervised model
Loss = []
for i in tqdm(range(n_epoch)):

    T0 = time.time()

    batch_state = data_bundle[g_i][0]  #TODO: rotate state
    batch_action = data_bundle[g_i][1].cuda()
    # batch_best_action = data_bundle[inner_i][2].cuda()
    target_Q = data_bundle[g_i][3].cuda()
    best_Q = data_bundle[g_i][4].cuda()

    # rotate state label

    rotate_operator = torch.nn.functional.one_hot(torch.tensor(template[np.random.randint(0, 6, num_state)].flatten()),
                                                  3).float().view(num_state, 3, 3).cuda()
    batch_state.ndata['label'] = torch.bmm(batch_state.ndata['label'].view(num_state, 9, 3), rotate_operator).view(-1, 3)

    S_a_encoding, h1, h2, Q_sa = model.forward_nognn(batch_state, batch_action)


    if loss_fc == 'pairwise':
        target_Q = Q_sa[0::2]
        best_Q = Q_sa[1::2]
        L = F.relu(target_Q - best_Q) # 2
    else:
        # L2-loss
        L = torch.pow(Q_sa - target_Q, 2)

        # state weighted L2
        # L = torch.pow(state_weight * (Q_sa - target_Q), 2)

        # weighted L2-loss
        # L = torch.pow(Q_sa - target_Q, 2) / (0.01 + best_Q - target_Q) \
        #  + 20 * F.relu(Q_sa - best_Q) # 3
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
               , np.round(T6-T0, 3)))
# with open('/p/reinforcement/data/gnn_rl/model/sup' + 'model_0', 'wb') as model_file:
#     pickle.dump(model, model_file)
# with open('/p/reinforcement/data/gnn_rl/model/sup' + 'model_0', 'rb') as model_file:
#     model = pickle.load(model_file)

# read model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + '/dqn_0129_adj_weight/'
with open(folder + 'dqn_' + str(10000), 'rb') as model_file:
    alg = pickle.load(model_file)
model = alg.model

num_state = 280 * 27
template = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])
all_state = set([state2QtableKey([int(i) for i in s[:-1].split(',')], reduce_rotate=True) for s in gen_comb012(3, 3, 3)])

for g_i in tqdm(range(100)):  # choose which graph instance to learn

    print('Analyze problem instance ', str(g_i), '...')

    # assign graph to instance
    g0 = dgl.unbatch(data_bundle[g_i][0])
    problem.g = g0[0]  # g0[i] differs in initial states

    # compute k-cut value at all different states
    for state in all_state:
        problem.reset_label(label=torch.tensor(QtableKey2state(state)).cuda())
        state_s_value[state] = problem.calc_S().item()
    # compute state_s_gain
    state_s_gain = {}.fromkeys(all_state)

    for state in all_state:
        #TODO: too slow!

        problem.reset_label(label=QtableKey2state(state))
        # problem.reset_label(label=[0,0,0,1,1,1,2,2,2])
        # rand_label = torch.tensor(range(3)).unsqueeze(1).expand(3, 3).flatten()[torch.randperm(9)]
        # problem.reset_label(label=rand_label)
        reward_seq = []
        action_seq = []
        diff_q = []
        for i in range(50):

            # find accurate q-values:
            state_loc = find_state(problem.g.ndata['label'].nonzero()[:, 1].cpu().numpy())
            acc_q = data_bundle[g_i][3][state_loc * 27: (state_loc+1) * 27]

            # predict q-values
            legal_actions = problem.get_legal_actions()
            _, _, _, Q_sa = model.forward(problem.g, legal_actions)

            # diff_q.append(torch.sum(torch.pow(Q_sa.cpu() - acc_q, 2)).item())

            best_actions = Q_sa.view(-1, 27).argmax(dim=1)
            action = legal_actions[best_actions]
            action = (action[0, 0].item(), action[0, 1].item())
            action_seq.append(action)
            _, reward = problem.step(action=action)
            reward_seq.append(reward.item())
        state_s_gain[state] = max(np.cumsum(reward_seq))# sum(reward_seq)
        # print(max(np.cumsum(reward_seq)))
    print('average gain:', str(sum(state_s_gain.values())/280))


    # plot S/gain for all states
    y = {}.fromkeys(all_state)

    for k in state_s_gain.keys():
        y[k] = (state_s_value[k], state_s_value[k]-state_s_gain[k])

    y = sorted(y.items(), key=lambda x: x[1][0], reverse=True)
    ys = [x[1][0] for x in y]
    yy = [x[1][1] for x in y]

    path = os.path.abspath(os.path.join(os.getcwd()))
    fig_name = 'case-study-max-gain-' + str(g_i)
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(111)
    ax.plot(ys, label='Initial K-cut value')
    ax.plot(yy, label='Best K-cut value reached in episode', color='r')
    ax.set_xlabel('States')
    ax.set_ylabel("K-cut value")
    ax.set_title('Best episode gain at different states')
    plt.legend(loc="upper right")
    plt.savefig(path + '/supervised/figs/' + fig_name + '.png')
    plt.close()


    # plot q-truth/q-pred for all states
    state_q_diff = {}.fromkeys(all_state)
    state_q_diff_m1 = {}.fromkeys(all_state)
    state_q_diff_m2 = {}.fromkeys(all_state)
    for state in all_state:

        problem.reset_label(label=QtableKey2state(state))

        # predict q-values
        legal_actions = problem.get_legal_actions()
        _, _, _, Q_sa = model.forward(problem.g, legal_actions)

        state_loc = find_state(problem.g.ndata['label'].nonzero()[:, 1].cpu().numpy())
        acc_q = data_bundle[g_i][3][state_loc * 27: (state_loc+1) * 27]

        # state_q_diff[state] = torch.sum(torch.pow(Q_sa.cpu() - acc_q, 2)).item()
        state_q_diff[state] = torch.std(Q_sa.cpu() - acc_q).item()
        state_q_diff_m1[state] = torch.mean(Q_sa.cpu()).item()
        state_q_diff_m2[state] = torch.mean(acc_q).item()
        print(state, state_q_diff[state])


    y = {}.fromkeys(all_state)

    for k in state_s_gain.keys():
        y[k] = (state_s_value[k], state_q_diff[k], state_q_diff_m1[k], state_q_diff_m2[k])

    y = sorted(y.items(), key=lambda x: x[1][0], reverse=True)
    ys = [x[1][0] for x in y]
    yy = [x[1][1] * 10 for x in y]
    yym1 = [x[1][2] for x in y]
    yym2 = [x[1][3] for x in y]

    path = os.path.abspath(os.path.join(os.getcwd()))
    fig_name = 'case-study-q-error-' + str(g_i)
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(111)
    ax.plot(ys, label='different state')
    ax.plot(yy, label='L2 error of q-prediction', color='r')
    ax.plot(yym1, label='L2 error of q-prediction', color='g')
    ax.plot(yym2, label='L2 error of q-prediction', color='b')
    ax.set_xlabel('States')
    ax.set_ylabel("L2 error of q-prediction")
    ax.set_title('Q value prediction error at different states')
    plt.legend(loc="upper right")
    plt.savefig(path + '/supervised/figs/' + fig_name + '.png')
    plt.close()
