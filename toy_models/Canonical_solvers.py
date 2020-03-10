import os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from DQN import DQN, to_cuda, EpisodeHistory
from k_cut import *
import numpy as np
import torch
from tqdm import tqdm
import pickle
from toy_models.Qiter import vis_g, state2QtableKey, QtableKey2state


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16, mode='complete')
res = np.zeros((3, 20000))
greedy_move_history = []
all_states = list(set([state2QtableKey(x) for x in list(itertools.permutations([0,0,0,1,1,1,2,2,2], 9))]))

select_problem = []

for i in range(20000):

    # # brute force
    problem.reset()
    res[0, i] = problem.calc_S().item()
    pb1 = dc(problem)
    pb1.g = to_cuda(pb1.g)
    S = []
    for j in range(280):
        pb1.reset_label(QtableKey2state(all_states[j]))
        S.append(pb1.calc_S())

    s1 = torch.tensor(S).argmin()
    pb1.reset_label(QtableKey2state(all_states[s1]))
    res[1, i] = S[s1]
    # print(s1)

    # greedy algorithm
    # problem.g = gg[i]
    pb2 = dc(problem)
    pb2.g = to_cuda(pb2.g)
    # pb2 = problem
    R = []
    Reward = []
    for j in range(100):
        M = []
        actions = pb2.get_legal_actions()
        for k in range(actions.shape[0]):
            _, r = pb2.step(actions[k], state=dc(pb2.g))
            M.append(r)
        if max(M) <= 0:
            break
        posi = [x for x in M if x > 0]
        nega = [x for x in M if x <= 0]
        # print('posi reward ratio:', len(posi) / len(M))
        # print('posi reward avg:', sum(posi) / len(posi))
        # print('nega reward avg:', sum(nega) / len(nega))
        max_idx = torch.tensor(M).argmax().item()
        _, r = pb2.step((actions[max_idx, 0].item(), actions[max_idx, 1].item()))
        R.append((actions[max_idx, 0].item(), actions[max_idx, 1].item(), r))
        Reward.append(r.item())
        # print('\n', j, Reward[-1])


        # path = os.path.abspath(os.path.join(os.getcwd())) + '/supervised/case_study/episode/in_vig56/' + str(i) + '/'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # vis_g(pb2, name=path + str(j), topo='knn-')

    greedy_move_history.append(Reward)
    s2 = pb2.calc_S().item()
    res[2, i] = s2
    # res[1, i] = len(Reward)
    # print('\n', res[:, i])

    # if res[1, i] < res[2, i] - 0.01:
    x = QtableKey2state(state2QtableKey(pb1.g.ndata['label'].argmax(dim=1).cpu().numpy()))
    y = QtableKey2state(state2QtableKey(pb2.g.ndata['label'].argmax(dim=1).cpu().numpy()))
    select_problem.append((dc(problem), x, y))
        # problem.reset_label(y)
        # problem.g = to_cuda(problem.g)
        # problem.calc_S()
    print(len(select_problem))
    if len(select_problem) == 100:
        break

path = '/p/reinforcement/data/gnn_rl/model/test_data/3by3/'
if not os.path.exists(path):
    os.makedirs(path)
with open(path + '1', 'wb') as model_file:
    pickle.dump(select_problem, model_file)
    # mean_start_s = sum(res[0,:]) / (i+1)
    # mean_best_s = sum(res[2, :]) / (i + 1)
    # mean_steps = sum(res[1, :]) / (i + 1)
    # mean_gain_s = mean_start_s - mean_best_s
    # mean_gain_ratio = mean_gain_s/mean_start_s
    #
    # print('Avg value of initial S:', mean_start_s)
    # print('Avg episode best value:', mean_best_s)
    # print('Avg step cost:', mean_steps)
    # print('Avg percentage max gain:', mean_gain_ratio)



# problem.g = to_cuda(select_problem[0][0].g)
# problem.reset_label(select_problem[0][1])
# problem.calc_S()
# batch_actions = problem.get_legal_actions()
# # problem.step((2, 5))
# # problem.step((0, 5))
#
# bg = dgl.batch([problem.g])
# S_a_encoding, h1, h2, Q_sa = alg.forward(bg, batch_actions, gnn_step=3)
#
# pb2 = dc(select_problem[0][0])
# pb2.g = to_cuda(pb2.g)
# pb2.reset_label(QtableKey2state(all_states[2]))
# for j in range(100):
#     M = []
#     actions = pb2.get_legal_actions()
#     for k in range(actions.shape[0]):
#         _, r = pb2.step(actions[k], state=dc(pb2.g))
#         M.append(r)
#     if max(M) <= 0:
#         break
#     posi = [x for x in M if x > 0]
#     nega = [x for x in M if x <= 0]
#     # print('posi reward ratio:', len(posi) / len(M))
#     # print('posi reward avg:', sum(posi) / len(posi))
#     # print('nega reward avg:', sum(nega) / len(nega))
#     max_idx = torch.tensor(M).argmax().item()
#     _, r = pb2.step((actions[max_idx, 0].item(), actions[max_idx, 1].item()))
#     R.append((actions[max_idx, 0].item(), actions[max_idx, 1].item(), r))
#     Reward.append(r.item())
#     # print('\n', j, Reward[-1])
#
#     # path = os.path.abspath(os.path.join(os.getcwd())) + '/supervised/case_study/episode/in_vig56/' + str(i) + '/'
#     # if not os.path.exists(path):
#     #     os.makedirs(path)
#     # vis_g(pb2, name=path + str(j), topo='knn-')
#
# greedy_move_history.append(Reward)
# pb2.calc_S().item()
#
# path = os.path.abspath(os.path.join(os.getcwd())) + '/supervised/case_study/test/opt1'
# if not os.path.exists(path):
#     os.makedirs(path)
# problem.g = to_cuda(select_problem[0][0].g)
# problem.reset_label(select_problem[0][1])
# vis_g(problem, name=path, topo='knn-')
