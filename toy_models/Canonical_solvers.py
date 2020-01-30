from DQN import DQN, to_cuda, EpisodeHistory
from k_cut import *
import numpy as np
import torch
from tqdm import tqdm
from toy_models.Qiter import vis_g, state2QtableKey, QtableKey2state

problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
res = np.zeros((3, 100))
greedy_move_history = []
for i in tqdm(range(100)):

    problem.reset()
    res[0, i] = problem.calc_S()
    pb1 = dc(problem)
    all_states = list(set([state2QtableKey(x) for x in list(itertools.permutations([0,0,0,1,1,1,2,2,2], 9))]))
    S = []
    for j in range(280):
        pb1.reset_label(QtableKey2state(all_states[j]))
        S.append(pb1.calc_S())

    s1 = min(S)
    res[1, i] = s1
    print(s1)

    pb2 = dc(problem)
    R = []
    Reward = []
    while True:
        M = []
        actions = pb2.get_legal_actions()
        for k in range(actions.shape[0]):
            _, r = pb2.step(actions[k], state=dc(pb2.g))
            M.append(r)
        if max(M) < 0:
            break
        max_idx = torch.tensor(M).argmax().item()
        _, r = pb2.step((actions[max_idx, 0].item(), actions[max_idx, 1].item()))
        R.append((actions[max_idx, 0].item(), actions[max_idx, 1].item(), r))
        Reward.append(r.item())

    greedy_move_history.append(Reward)
    s2 = pb2.calc_S()
    res[2, i] = s2
    print(s2)

    # if not s1 == s2:
    #     break

pb = dc(problem)
vis_g(pb, name='toy_models/0114_0', topo='c')
for i in range(len(R)):
    pb.step(R[i][0:2])
    vis_g(pb, name='toy_models/0114_' + str(i+1), topo='c')


pb.reset_label([0,2,2,0,2,0,1,1,1])
