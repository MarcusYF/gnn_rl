from DQN import *
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Qiter_swap import state2QtableKey, QtableKey2state, gen_comb012
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
action_type = 'swap'

k = 3
m = 3
ajr = 5
hidden_dim = 16
extended_h = True
use_x = False
lr = 1e-4
n_epoch = 10000

# read data
with open('/p/reinforcement/data/gnn_rl/sup_split_100graphs/batch_0', 'rb') as f:
    data_bundle = pickle.load(f)

def find_state(label):
    # find the location for each state
    for i in range(280):
        tmp = data_bundle[g_i][0].ndata['label'][i*9*27:i*9*27+9].nonzero()[:, 1].cpu().numpy()
        if state2QtableKey(tmp) == state2QtableKey(label):
            return i

def cal_ndcg(Q_sa, acc_q, k):
    # acc_q = 1 / (torch.exp(a * acc_q) + 1)
    # acc_q = acc_q - min(acc_q)
    action_order = Q_sa.argsort(descending=True)
    action_preference_seq = acc_q[action_order]
    action_true_importance_seq = acc_q.sort(descending=True).values
    dcg_weight = [np.log(2) / np.log(i + 2) for i in range(k)]

    return sum([action_preference_seq[i].item() * dcg_weight[i] for i in range(k)]) / sum([action_true_importance_seq[i].item() * dcg_weight[i] for i in range(k)])

# read model
folder = '/p/reinforcement/data/gnn_rl/model/dqn/' + '/dqn_0129_adj_weight/'
with open(folder + 'dqn_' + str(10000), 'rb') as model_file:
    alg = pickle.load(model_file)
model = alg.model

num_state = 280 * 27
template = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])
all_state = set([state2QtableKey([int(i) for i in s[:-1].split(',')], reduce_rotate=True) for s in gen_comb012(3, 3, 3)])

problem = KCut_DGL(k=k, m=m, adjacent_reserve=ajr, hidden_dim=hidden_dim)
result_accumulator = []
for g_i in tqdm(range(100)):  # choose which graph instance to learn

    print('\nAnalyze problem instance ', str(g_i), '...')

    # assign graph to instance
    g0 = dgl.unbatch(data_bundle[g_i][0])
    problem.g = g0[0]  # g0[i] differs in initial states

    # compute k-cut value at all different states
    state_s_value = {}.fromkeys(all_state)
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
    plt.savefig(path + '/supervised/case_study/' + fig_name + '.png')
    plt.close()


    # plot q-truth/q-pred for all states
    state_q_diff = {}.fromkeys(all_state)
    state_q_diff_m1 = {}.fromkeys(all_state)
    state_q_diff_m2 = {}.fromkeys(all_state)
    state_q_diff_ndcg1 = {}.fromkeys(all_state)
    state_q_diff_ndcg5 = {}.fromkeys(all_state)
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
        state_q_diff_ndcg1[state] = cal_ndcg(Q_sa, acc_q, 1)
        state_q_diff_ndcg5[state] = cal_ndcg(Q_sa, acc_q, 5)



    y = {}.fromkeys(all_state)
    z = {}.fromkeys(all_state)
    for k in state_s_gain.keys():
        y[k] = (state_s_value[k], state_s_gain[k], state_q_diff[k], state_q_diff_m1[k], state_q_diff_m2[k])
        z[k] = (state_s_value[k], state_q_diff_ndcg1[k], state_q_diff_ndcg5[k])

    y = sorted(y.items(), key=lambda x: x[1][0], reverse=True)
    z = sorted(z.items(), key=lambda x: x[1][0], reverse=True)
    ys = [x[1][0] for x in y]
    yy = [x[1][2] * 10 for x in y]
    yym1 = [x[1][3] for x in y]
    yym2 = [x[1][4] for x in y]

    z1 = [x[1][1] if x[1][1] <= 1 else 0.6 for x in z]
    z5 = [x[1][2] if x[1][2] <= 1 else 0.6 for x in z]

    path = os.path.abspath(os.path.join(os.getcwd()))
    fig_name = 'case-study-q-error-' + str(g_i)
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(111)

    ax2 = ax.twinx()
    ax2.plot(z1, label='NDCG@1 for predicted q-value')
    ax2.plot(z5, label='NDCG@5 for predicted q-value')
    ax2.set_ylabel("NDCG")

    ax.plot(ys, label='Initial K-cut value', color='r')
    # ax.plot(yy, label='L2 error of predicted q-value', color='r')
    # ax.plot(yym1, label='Avg. predicted q-value', color='g')
    # ax.plot(yym2, label='Avg. ground truth q-value', color='b')
    ax.set_xlabel('States')
    ax.set_ylabel("K-cut value")
    ax.set_title('Q-value prediction quality at different states')

    plt.legend(loc="lower left")
    plt.savefig(path + '/supervised/case_study/ndcg/' + fig_name + '.png')
    plt.close()

    result_accumulator.append(y)

# plot q-value error bars
g_i = 1
fontsize=30
max_q = data_bundle[g_i][3].view(280, 27).max(dim=1).values
min_q = data_bundle[g_i][3].view(280, 27).min(dim=1).values
avg_q = data_bundle[g_i][3].view(280, 27).mean(dim=1)
a=avg_q.sort(descending=True).indices
path = os.path.abspath(os.path.join(os.getcwd()))

fig_name = 'q-error-bar-' + str(g_i)
fig = plt.figure(figsize=[15, 15])
ax = fig.add_subplot(111)
ax.errorbar(x=list(range(0,280)), y=avg_q[a][0:280], yerr=torch.cat([avg_q[a] - min_q[a], max_q[a]-avg_q[a]]).view(2,-1)[:,0:280], label='avg. q-value')
# ax.errorbar(x=list(range(0,280)), y=np.zeros(280), yerr=torch.cat([avg_q[a] - min_q[a], max_q[a]-avg_q[a]]).view(2,-1)[:,0:280]
#             , label='avg. q-value')
ax.set_ylabel("Ground truth q-value", fontsize=fontsize)
ax.set_xlabel('States', fontsize=fontsize)
ax.set_title('Q-value error bar', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc="lower left", fontsize=fontsize)
plt.savefig(path + '/supervised/case_study/ndcg/' + fig_name + '.png')
plt.close()

fig_name = 'q-error-bar-flatten-' + str(g_i)
fig = plt.figure(figsize=[15, 15])
ax = fig.add_subplot(111)
# ax.errorbar(x=list(range(0,280)), y=avg_q[a][0:280], yerr=torch.cat([avg_q[a] - min_q[a], max_q[a]-avg_q[a]]).view(2,-1)[:,0:280], label='avg. q-value')
ax.errorbar(x=list(range(0,280)), y=np.zeros(280), yerr=torch.cat([avg_q[a] - min_q[a], max_q[a]-avg_q[a]]).view(2,-1)[:,0:280], label='avg. q-value shift to zero')
ax.set_ylabel("Ground truth q-value", fontsize=fontsize)
ax.set_xlabel('States', fontsize=fontsize)
ax.set_title('Q-value error bar', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc="lower left", fontsize=fontsize)
plt.savefig(path + '/supervised/case_study/ndcg/' + fig_name + '.png')
plt.close()

fig_name = 'q-error-bar-relative-' + str(g_i)
fig = plt.figure(figsize=[15, 15])
ax = fig.add_subplot(111)
# ax.errorbar(x=list(range(0,280)), y=avg_q[a][0:280], yerr=torch.cat([avg_q[a] - min_q[a], max_q[a]-avg_q[a]]).view(2,-1)[:,0:280], label='avg. q-value')
ax.errorbar(x=list(range(0,280)), y=np.zeros(280), yerr=torch.cat([(avg_q[a] - min_q[a])/avg_q[a], (max_q[a]-avg_q[a])/avg_q[a]]).view(2,-1)[:,0:280], label='avg. q-value shift to zero')
ax.set_ylabel("Ground truth q-value", fontsize=fontsize)
ax.set_xlabel('States', fontsize=fontsize)
ax.set_title('Q-value error bar', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc="lower left", fontsize=fontsize)
plt.savefig(path + '/supervised/case_study/ndcg/' + fig_name + '.png')
plt.close()
