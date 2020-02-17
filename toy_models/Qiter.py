from k_cut import KCut_DGL
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import torch as th
import itertools
from tqdm import tqdm
import numpy as np
import pickle

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
    X = g.ndata['x'].cpu()
    n = X.shape[0]
    label = g.ndata['label'].cpu()
    link = dc(g.edata['e_type'].view(n, n - 1, 2).cpu())
    # c = ['r', 'b', 'y']
    plt.cla()
    c = ['r', 'b', 'y', 'k', 'g', 'c', 'm', 'tan', 'peru', 'pink']
    for i in range(k):
        a = X[(label[:, i] > 0).nonzero().squeeze()]
        if a.shape[0] > .5:
            a = a.view(-1, 2)
            plt.scatter(a[:, 0], a[:, 1], s=60, c=[c[i]]*(n//k))

    for i in range(n):
        plt.annotate(str(i), xy=(X[i, 0], X[i, 1]))
        for j in range(n - 1):
            if link[i, j][0].item() == 1:
                j_ = j
                if j >= i:
                    j_ = j + 1
                plt.plot([X[i, 0], X[j_, 0]], [X[i, 1], X[j_, 1]], ':', color='k')

        if topo == 'cut':
            for j in range(n - 1):
                if link[i, j][1].item() + link[i, j][0].item() > 1.5:
                    j_ = j
                    if j >= i:
                        j_ = j + 1
                    plt.plot([X[i, 0], X[j_, 0]], [X[i, 1], X[j_, 1]], '-', color='k')


    plt.savefig(name + '.png')
    plt.close()

if __name__ == '__main__':
    n, k, m = 9, 3, 3
    problem = KCut_DGL(k=k, m=m, adjacent_reserve=5, hidden_dim=8)
    #
    # x = th.tensor([[0.1,0.1],[0.1,0.2],[0.2,0.1],[0.2,0.2],[0.12,0.23],[0.9,0.9],[0.8,0.8],[0.9,0.8],[0.8,0.9]])
    # problem = KCut_DGL(k=k, m=m, adjacent_reserve=5, hidden_dim=8, random_sample_node=False, x=x)
    # # problem = KCut_DGL(k=k, m=m, adjacent_reserve=5, hidden_dim=8, random_init_label=False, random_sample_node=False
    # #                    , label=[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
    # #                    , x=th.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [0.7, 0.8], [1.5, 1.5], [1.5, 3], [3, 3], [3, 1.5], [1.8, 1.7], [1.8, 0.3], [1.8, 0.8], [2.3, 0.8], [2.3, 0.3], [2.1, 0.5]]))
    #
    # problem.calc_S()
    # problem.g.ndata['label']
    # vis_g(problem, name='toy_models/toy93', topo='c')
    # vis_g(problem, name='toy_models/toy94')
    # problem.reset_label([0,1,1,0,1,2,2,0,2])
    # problem.calc_S()
    # problem.step((1,2), action_type='flip')
    # problem.calc_S()
    # problem.g.ndata['label']
    # vis_g(problem, name='toy_models/flip0', topo='cluster')
    # # g = generate_G(k=k, m=4, adjacent_reserve=1, hidden_dim=8, random_init_label=True, a=1, sample=False)['g']



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

    folder = '/u/fy4bc/code/research/RL4CombOptm/gnn_rl/Data/qiter33/'
    # folder = 'Models/dqn_test_not_sample_batch_episode/'
    with open(folder + 'qtable_chunk_0_0', 'rb') as data_file:
        data = pickle.load(data_file)

    # test
    Q_table = Q_table_280
    episode_name = 'test2_swap_'
    Obj = []
    Reward = []
    init_solution = np.random.permutation([1,1,1,2,2,2,0,0,0])
    problem.reset_label(init_solution)
    vis_g(problem, name=episode_name+'0', topo='cluster')
    Obj.append(problem.calc_S())

    ite = 1
    solution = state2QtableKey(init_solution)
    while ite < 20 and np.max(Q_table[solution]) > 0:
        print(Q_table[solution])
        action = np.argmax(Q_table[solution])
        if action_type == 'swap':
            action = action // n, action % n
        if action_type == 'flip':
            action = action // k, action % k

        _, reward = problem.step(action, action_type=action_type)
        solution = QtableKey2state(solution)

        if action_type == 'swap':
            solution[action[0]], solution[action[1]] = solution[action[1]], solution[action[0]]
        if action_type == 'flip':
            solution[action[0]] = action[1]

        solution = state2QtableKey(solution)
        Obj.append(problem.calc_S())
        Reward.append(reward)
        reward
        vis_g(problem, name=episode_name+str(ite), topo='cluster')
        ite += 1
    Obj


