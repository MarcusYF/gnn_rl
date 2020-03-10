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
from copy import deepcopy as dc
import matplotlib.pyplot as plt
from Qiter_swap import state2QtableKey, QtableKey2state, gen_comb012




all_state = set([state2QtableKey([int(i) for i in s[:-1].split(',')], reduce_rotate=True) for s in gen_comb012(3, 3, 3)])


state_visit_count = {}.fromkeys(all_state, 0)
begin_state = QtableKey2state('0,1,2,2,1,1,2,0,0')
for k in tqdm(range(1000000)):
    i, j = random.sample([0,1,2,3,4,5,6,7,8], 2)
    state_ = dc(begin_state)
    begin_state[i], begin_state[j] = begin_state[j], begin_state[i]
    new_ = state2QtableKey(begin_state)
    old_ = state2QtableKey(state_)
    if new_ == old_:
        continue
    else:
        state_visit_count[new_] += 1


n, k, m = 9, 3, 3
problem = KCut_DGL(k=k, m=m, adjacent_reserve=5, hidden_dim=8)

action_type = 'swap'

if action_type == 'swap':
    all_state = set([state2QtableKey(x, reduce_rotate=False) for x in list(itertools.permutations([0,0,0,1,1,1,2,2,2], n))])

if action_type == 'flip':
    all_state = [state2QtableKey(x) for x in list(itertools.product([0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1,2]))]

Q_table = {}
for state in tqdm(all_state):
    if state in Q_table.keys():
        continue
    l = QtableKey2state(state)
    if action_type == 'swap':
        R = []
        for i, j in itertools.product(range(n), range(n)):
            problem.reset_label(l, calc_S=False, rewire_edges=False)
            _, r = problem.step((i, j), rewire_edges=False, action_type='swap')
            R.append(r.item())
        Q_table[state] = np.array(R)
    if action_type == 'flip':
        R = np.zeros([n, k])
        for i, target_label in itertools.product(range(n), range(k)):
            problem.reset_label(l, calc_S=False, rewire_edges=False)

            old_balance = problem.g.ndata['label'].sum(dim=0)

            _, r = problem.step((i, target_label), rewire_edges=False, action_type='flip')

            new_balance = problem.g.ndata['label'].sum(dim=0)

            cluster_balance_penalty = (max(new_balance) - min(new_balance)) - (max(old_balance) - min(old_balance))

            R[i, target_label] = r.item() - 1.0 * cluster_balance_penalty

        Q_table[state] = R


# Start Q-value iteration
problem_bak = dc(problem) # detached backup for problem instance
Q_table_bak = dc(Q_table) # detached backup for Q-table


Q_table_ = dc(Q_table_bak) # target Q-table
Q_table = dc(Q_table_bak) # Q-table
State_Action_Reward = dc(Q_table_bak)
gamma = 0.90



Err = []
for ite in tqdm(range(10)):
    for state in Q_table.keys():
        if action_type == 'swap':
            for i, j in itertools.product(range(n), range(n)):
                action = n*i+j
                state_ = QtableKey2state(state)
                state_[i], state_[j] = state_[j], state_[i]  # swap
                new_state = state2QtableKey(state_)
                Q_table[state][action] = State_Action_Reward[state][action] + gamma * max(Q_table_[new_state])
        if action_type == 'flip':
            for i, target_label in itertools.product(range(n), range(k)):
                action = i, target_label
                state_ = QtableKey2state(state)
                state_[i] = target_label  # flip
                new_state = state2QtableKey(state_)
                Q_table[state][action] = State_Action_Reward[state][action] + gamma * np.max(Q_table_[new_state])
    # update target Q-table
    err = 0
    for state in Q_table.keys():
        diff = Q_table[state] - Q_table_[state]
        err += np.sqrt(np.mean(diff ** 2))
    Err.append(err)
    print(err)
    Q_table_ = dc(Q_table)

g_i = 0

g0 = dgl.unbatch(data_bundle[g_i][0])
problem.g = g0[0]  # g0[i] differs in initial states

# compute k-cut value at all different states
state_s_value = {}.fromkeys(all_state)
for state in all_state:
    problem.reset_label(label=torch.tensor(QtableKey2state(state)).cuda())
    state_s_value[state] = problem.calc_S().item()

l = sorted(state_s_value.items(),key=lambda x:x[1])
best_label = [int(x) for x in l[0][0].split(',')]


i = find_state(best_label)

problem.reset_label(label=best_label)
problem.calc_S()
actions = problem.get_legal_actions()
for i, j in actions:
    print('action:', i, j)
    new_label = dc(best_label)
    new_label[i], new_label[j] = new_label[j], new_label[i]
    problem.reset_label(label=new_label)
    print(problem.calc_S())

    kk = find_state(new_label)
    print(data_bundle[g_i][3][27*kk:(kk+1)*27])
    print('best_q_action:', problem.get_legal_actions()[torch.argmax(data_bundle[g_i][3][27*kk:(kk+1)*27])])
    # state2QtableKey(new_label)


