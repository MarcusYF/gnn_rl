from DQN import DQN, to_cuda, EpisodeHistory
from k_cut import *
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm
from toy_models.Qiter import vis_g, state2QtableKey, QtableKey2state
from pandas import DataFrame
import dgl


class test_summary:

    def __init__(self, alg, problem=None, q_net='mlp', forbid_revisit=False):
        if isinstance(alg, DQN):
            self.alg = alg.model
        else:
            self.alg = alg

        self.problem = problem
        if isinstance(problem.g, BatchedDGLGraph):
            self.n = problem.g.ndata['label'].shape[0] // problem.g.batch_size
        else:
            self.n = problem.g.ndata['label'].shape[0]
        # if isinstance(problem, BatchedDGLGraph):
        #     self.problem = problem
        # elif isinstance(problem, list):
        #     self.problem = dgl.batch(problem)
        # else:
        #     self.problem = False

        self.episodes = []
        self.S = []
        self.max_gain = []
        self.max_gain_budget = []
        self.max_gain_ratio = []
        # self.action_indices = DataFrame(range(27))
        self.q_net = q_net
        self.forbid_revisit = forbid_revisit

        self.all_states = list(set([state2QtableKey(x) for x in list(itertools.permutations([0, 0, 0, 1, 1, 1, 2, 2, 2], 9))]))

        self.state_eval = []

    # need to adapt from Canonical_solvers.py
    def cmpt_optimal(self, graph, path=None):

        self.problem.g = to_cuda(graph)
        res = [self.problem.calc_S().item()]
        pb = self.problem

        S = []
        for j in range(280):
            pb.reset_label(QtableKey2state(self.all_states[j]))
            S.append(pb.calc_S())

        s1 = torch.tensor(S).argmin()
        res.append(S[s1].item())

        if path is not None:
            path = os.path.abspath(os.path.join(os.getcwd())) + path
            pb.reset_label(QtableKey2state(self.all_states[s1]))
            vis_g(pb, name=path, topo='cut')
        return QtableKey2state(self.all_states[s1]), res

    # need to adapt from Canonical_solvers.py
    def test_greedy(self, graph, path=None):
        self.problem.g = to_cuda(graph)
        res = [self.problem.calc_S().item()]
        pb = self.problem
        if path is not None:
            path = os.path.abspath(os.path.join(os.getcwd())) + path
            vis_g(pb, name=path + str(0), topo='cut')
        R = []
        Reward = []
        for j in range(100):
            M = []
            actions = pb.get_legal_actions()
            for k in range(actions.shape[0]):
                _, r = pb.step(actions[k], state=dc(pb.g))
                M.append(r)
            if max(M) <= 0:
                break
            if path is not None:
                vis_g(pb, name=path + str(j+1), topo='cut')
            posi = [x for x in M if x > 0]
            nega = [x for x in M if x <= 0]
            # print('posi reward ratio:', len(posi) / len(M))
            # print('posi reward avg:', sum(posi) / len(posi))
            # print('nega reward avg:', sum(nega) / len(nega))
            max_idx = torch.tensor(M).argmax().item()
            _, r = pb.step((actions[max_idx, 0].item(), actions[max_idx, 1].item()))
            R.append((actions[max_idx, 0].item(), actions[max_idx, 1].item(), r.item()))
            Reward.append(r.item())
            res.append(res[-1] - r.item())
        return QtableKey2state(state2QtableKey(pb.g.ndata['label'].argmax(dim=1).cpu().numpy())), R, res
