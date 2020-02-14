import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from DQN import DQN, to_cuda, EpisodeHistory
from k_cut import *
import numpy as np
import torch
from tqdm import tqdm
from toy_models.Qiter import vis_g, state2QtableKey, QtableKey2state


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

problem = KCut_DGL(k=10, m=10, adjacent_reserve=20, hidden_dim=64, mode='complete')
res = np.zeros((3, 100))
greedy_move_history = []
all_states = list(set([state2QtableKey(x) for x in list(itertools.permutations([0,0,0,1,1,1,2,2,2], 9))]))
for i in tqdm(range(1)):

    # # brute force
    problem.reset()
    res[0, i] = problem.calc_S().item()
    # pb1 = dc(problem)
    # pb1.g = to_cuda(pb1.g)
    # S = []
    # for j in range(280):
    #     pb1.reset_label(QtableKey2state(all_states[j]))
    #     S.append(pb1.calc_S())
    #
    # s1 = min(S)
    # res[1, i] = s1
    # print(s1)

    # greedy algorithm
    # problem.g = gg[i]
    pb2 = dc(problem)
    pb2.g = to_cuda(pb2.g)
    # pb2 = problem
    R = []
    Reward = []
    for j in tqdm(range(100)):
        M = []
        actions = pb2.get_legal_actions()
        for k in range(actions.shape[0]):
            _, r = pb2.step(actions[k], state=dc(pb2.g))
            M.append(r)
        if max(M) < 0:
            break
        posi = [x for x in M if x > 0]
        nega = [x for x in M if x <= 0]
        print('posi reward ratio:', len(posi) / len(M))
        print('posi reward avg:', sum(posi) / len(posi))
        print('nega reward avg:', sum(nega) / len(nega))
        max_idx = torch.tensor(M).argmax().item()
        _, r = pb2.step((actions[max_idx, 0].item(), actions[max_idx, 1].item()))
        R.append((actions[max_idx, 0].item(), actions[max_idx, 1].item(), r))
        Reward.append(r.item())
        # print('\n', j, Reward[-1])


        path = os.path.abspath(os.path.join(os.getcwd())) + '/supervised/case_study/episode/in_vig56/' + str(i) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        vis_g(pb2, name=path + str(j), topo='knn-')

    greedy_move_history.append(Reward)
    s2 = pb2.calc_S().item()
    res[2, i] = s2
    res[1, i] = len(Reward)
    # print('\n', res[:,i])

    mean_start_s = sum(res[0,:]) / (i+1)
    mean_best_s = sum(res[2, :]) / (i + 1)
    mean_steps = sum(res[1, :]) / (i + 1)
    mean_gain_s = mean_start_s - mean_best_s
    mean_gain_ratio = mean_gain_s/mean_start_s

    print('Avg value of initial S:', mean_start_s)
    print('Avg episode best value:', mean_best_s)
    print('Avg step cost:', mean_steps)
    print('Avg percentage max gain:', mean_gain_ratio)





# pb = dc(problem)
# vis_g(pb, name='toy_models/0114_0', topo='c')
# for i in range(len(R)):
#     pb.step(R[i][0:2])
#     vis_g(pb, name='toy_models/0114_' + str(i+1), topo='c')
#
#
# pb.reset_label([0,2,2,0,2,0,1,1,1])
