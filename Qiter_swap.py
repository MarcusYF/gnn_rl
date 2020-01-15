from k_cut import KCut_DGL
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import os
import pickle
import itertools
from tqdm import tqdm
import numpy as np
import argparse

def state2QtableKey(s, reduce_rotate=True):
    # ruduce state space
    if reduce_rotate:
        first_ergodic = []
        for i in s:
            if i not in first_ergodic:
                first_ergodic.append(i)
        m = dict(zip(first_ergodic, range(len(first_ergodic))))
        s = [m[i] for i in s]
    return ','.join([str(x) for x in s])

def QtableKey2state(k):
    return [int(x) for x in k.split(',')]

def vis_g(problem, name='test', topo='knn'):
    k = problem.k
    g = problem.g
    X = g.ndata['x']
    n = X.shape[0]
    label = g.ndata['label']
    link = dc(g.edata['e_type'].view(n, n - 1, 2))
    c = ['r', 'b', 'y']
    plt.cla()

    for i in range(k):
        a = X[(label[:, i] > 0).nonzero().squeeze()]
        if a.shape[0] > .5:
            a = a.view(-1, 2)
            plt.scatter(a[:, 0], a[:, 1], s=60, c=c[i])

    for i in range(n):
        plt.annotate(str(i), xy=(X[i, 0], X[i, 1]))
        for j in range(n - 1):
            if topo=='knn':
                topo = 0
            else:
                topo = 1
            if link[i, j][topo].item() == 1:
                j_ = j
                if j >= i:
                    j_ = j + 1
                plt.plot([X[i, 0], X[j_, 0]], [X[i, 1], X[j_, 1]], '-', color='k')

    plt.savefig('./' + name + '.png')
    plt.close()

def gen_q_table(problem_instance, max_ite=200, err_bound=1e-3):

    problem = dc(problem_instance)

    n, k, m = problem.N, problem.k, problem.m
    labels = [0,0,0,1,1,1,2,2,2]
    all_state = set([state2QtableKey(x, reduce_rotate=True) for x in list(itertools.permutations(labels, n))])

    Q_table = {}
    for state in all_state:
        if state in Q_table.keys():
            continue
        Q_table[state] = {}
        l = QtableKey2state(state)
        R = []

        problem.reset_label(l, calc_S=False, rewire_edges=False)
        actions = problem.get_legal_actions()
        for ac in actions:
            i, j = ac.numpy()[0], ac.numpy()[1]
            _, r = problem.step((i, j), rewire_edges=False, action_type='swap')
            problem.reset_label(l, calc_S=False, rewire_edges=False)
            Q_table[state][(i, j)] = r


    # Start Q-value iteration
    problem_bak = dc(problem) # detached backup for problem instance
    Q_table_bak = dc(Q_table) # detached backup for Q-table


    Q_table_ = dc(Q_table_bak) # target Q-table
    Q_table = dc(Q_table_bak) # Q-table
    State_Action_Reward = dc(Q_table_bak)
    gamma = 0.90



    Err = []
    action_len = 27
    for ite in range(max_ite):
        for state in Q_table.keys():
            for action in Q_table[state].keys():
                i, j = action
                state_ = QtableKey2state(state)
                state_[i], state_[j] = state_[j], state_[i]  # swap to new state
                new_state = state2QtableKey(state_)
                Q_table[state][action] = State_Action_Reward[state][action] + gamma * max(Q_table_[new_state].values())
            # update target Q-table
        err = 0
        for state in Q_table.keys():
            new_v = [v for v in Q_table[state].values()]
            old_v = [v for v in Q_table_[state].values()]
            diff = 0
            for i in range(action_len):
                diff += (new_v[i] - old_v[i]) ** 2
            err += np.sqrt(diff / action_len)
        Err.append(err)

        Q_table_ = dc(Q_table)
        if err < err_bound:
            break

    for k in Q_table_.keys():
        for a in Q_table_[k]:
            Q_table[k][a] = (State_Action_Reward[k][a].item(), Q_table[k][a].item())
    State_Action_Reward, Q_table

    print('Q-value iteration exits with iterations: {}, error: {}'.format(ite, err.item()))

    return Q_table, err.item()

    # # test
    # Q_table
    # episode_name = 'test3_swap_'
    # Obj = []
    # Reward = []
    # init_solution = np.random.permutation([1,1,1,2,2,2,0,0,0])
    # problem.reset_label(init_solution)
    # vis_g(problem, name=episode_name+'0', topo='cluster')
    # Obj.append(problem.calc_S())
    #
    # ite = 1
    # solution = state2QtableKey(init_solution)
    # action, qv = sorted(Q_table[solution].items(), key=lambda x: x[1], reverse=True)[0]
    # while ite < 20 and qv > 0:
    #     print(Q_table[solution])
    #     _, reward = problem.step(action)
    #     solution = QtableKey2state(solution)
    #     solution[action[0]], solution[action[1]] = solution[action[1]], solution[action[0]]
    #     solution = state2QtableKey(solution)
    #     action, qv = sorted(Q_table[solution].items(), key=lambda x: x[1], reverse=True)[0]
    #     Obj.append(problem.calc_S())
    #     Reward.append(reward)
    #     reward
    #     vis_g(problem, name=episode_name+str(ite), topo='cluster')
    #     ite += 1
    # Obj

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate groud truth Q-table")
    parser.add_argument('--save_folder', default='qiter33')
    parser.add_argument('--gpu', default='0', help="")
    parser.add_argument('--batch_size', default=100, help="")

    args = vars(parser.parse_args())

    save_folder = args['save_folder']
    gpu = args['gpu']
    batch_size = int(args['batch_size'])

    path = 'Data/' + save_folder + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=8)
    for i in tqdm(range(batch_size)):
        q_table_batch = []
        for j in range(100):
            problem.reset()
            Q_table, err = gen_q_table(problem)
            q_table_batch.append((dc(problem), Q_table, err))
        with open(path + 'qtable_chunk_' + gpu + '_' + str(i), 'wb') as data_file:
            pickle.dump(q_table_batch, data_file)

