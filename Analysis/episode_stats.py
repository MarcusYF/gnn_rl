from DQN import DQN, to_cuda, EpisodeHistory
from k_cut import *
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import torch
from tqdm import tqdm
from toy_models.Qiter import vis_g
from pandas import DataFrame
import dgl


class test_summary:

    def __init__(self, alg, problem=None, q_net='mlp'):
        if isinstance(alg, DQN):
            self.alg = alg.model
        else:
            self.alg = alg

        self.problem = problem
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

    # need to adapt from Canonical_solvers.py
    def cmpt_optimal(self):
        return

    # need to adapt from Canonical_solvers.py
    def test_greedy(self):
        return

    def run_test(self, problem=None, trial_num=1, batch_size=100, gnn_step=3, episode_len=50, explore_prob=0.1, Temperature=1.0):
        self.num_actions = self.problem.N * (self.problem.N - self.problem.m) // 2

        if problem is None:
            self.batch_size = batch_size
            batch_size *= trial_num
            self.trial_num = trial_num
            bg = self.problem.gen_batch_graph(batch_size=self.batch_size)
            self.bg = to_cuda(self.problem.gen_batch_graph(x=bg.ndata['x'].repeat(trial_num, 1), batch_size=batch_size))
            gl = dgl.unbatch(self.bg)
            test_problem = self.problem
        else:
            # unit test
            batch_size = 1
            self.batch_size = batch_size
            self.bg = dgl.batch([problem.g])
            gl = dgl.unbatch(self.bg)
            self.problem = dc(problem)
            test_problem = self.problem

        self.action_mask = torch.tensor(range(0, self.num_actions * batch_size, self.num_actions)).cuda()

        self.S = [test_problem.calc_S(g=gl[i]).cpu() for i in range(batch_size)]

        ep = [EpisodeHistory(gl[i], max_episode_len=episode_len, action_type='swap') for i in range(batch_size)]
        self.end_of_episode = {}.fromkeys(range(batch_size), episode_len-1)
        for i in tqdm(range(episode_len)):
            batch_legal_actions = test_problem.get_legal_actions(state=self.bg, action_type='swap').cuda()

            if self.q_net == 'mlp':
                S_a_encoding, h1, h2, Q_sa = self.alg.forward(self.bg, batch_legal_actions, gnn_step=gnn_step)
            else:
                S_a_encoding, h1, h2, Q_sa = self.alg.forward_MHA(self.bg, batch_legal_actions, gnn_step=gnn_step)
            # # epsilon greedy strategy
            # if torch.rand(1) > explore_prob:
            #     # action_idx1 = Q_sa.argmax()
            #     weight = torch.softmax(softmax_constant * Q_sa.detach().cpu(), dim=0)
            #     # print(max(weight))
            #     action_idx = self.action_indices.sample(n=1, weights=weight).values[0][0]
            #     # print('111', action_idx1)
            #     # print('222', action_idx)
            # else:
            #     action_idx = torch.randint(high=legal_actions.shape[0], size=(1,)).squeeze()

            # print(Q_sa.view(-1, self.num_actions).max(dim=1))
            # best_actions = Q_sa.view(-1, self.num_actions).argmax(dim=1)
            # print(Q_sa.view(-1, self.num_actions).max(dim=1).values)
            terminate_episode = (Q_sa.view(-1, self.num_actions).max(dim=1).values < 0.).nonzero().flatten().cpu().numpy()
            # print(terminate_episode)
            for idx in terminate_episode:
                if self.end_of_episode[idx] == episode_len - 1:
                    self.end_of_episode[idx] = i

            best_actions = torch.multinomial(F.softmax(Q_sa.view(-1, self.num_actions) / Temperature), 1).view(-1)

            chose_actions = torch.tensor(
                [x if torch.rand(1) > explore_prob else torch.randint(high=self.num_actions, size=(1,)).squeeze() for x in
                 best_actions]).cuda()
            chose_actions += self.action_mask

            actions = batch_legal_actions[chose_actions]

            _, rewards = self.problem.step_batch(states=self.bg, action=actions)

            R = [reward.item() for reward in rewards]

            # print(R[0])
            # print(- torch.sum(F.softmax(Q_sa) * F.log_softmax(Q_sa)).cpu().item())
            # print(torch.max(Q_sa).cpu().item(), torch.mean(Q_sa).cpu().item(), torch.min(Q_sa).cpu().item(), torch.std(Q_sa).cpu().item())
            # print(torch.std(Q_sa))
            [ep[k].write(action=actions[k, :], action_idx=best_actions[k] - self.action_mask[k], reward=R[k]) for k in range(batch_size)]

        self.episodes = ep

    def show_result(self):
        # print(self.end_of_episode)
        initial_S = []
        episode_gains = []
        episode_max_gains = []
        episode_gain_ratios = []
        episode_max_gain_ratios = []
        # select best result among trials with different initialisations
        for i in range(self.batch_size * self.trial_num):
            episode_max_gains.append(max(max(np.cumsum(self.episodes[i].reward_seq)), 0))
            episode_max_gain_ratios.append(max(episode_max_gains[-1], 0) / self.S[i].item())
        bs = self.batch_size
        x = [bs * np.argmax(episode_max_gain_ratios[i::bs]) for i in range(bs)]
        select_indices = [x[i] + i for i in range(bs)]
        episode_max_gains = []
        episode_max_gain_ratios = []
        for i in select_indices:
            initial_S.append(self.S[i].item())
            episode_gains.append(self.S[i].item() - sum(self.episodes[i].reward_seq))
            episode_max_gains.append(self.S[i].item() - max(max(np.cumsum(self.episodes[i].reward_seq)), 0))
            episode_gain_ratios.append(episode_gains[-1] / self.S[i].item())
            episode_max_gain_ratios.append(max(episode_max_gains[-1], 0) / self.S[i].item())
        print('Avg value of initial S:', np.mean(self.S))
        print('Avg episode end value:', np.mean(episode_gains))
        print('Avg episode best value:', np.mean(episode_max_gains))
        # print('Avg max gain budget:', np.mean(self.max_gain_budget))
        # print('Var max gain budget:', np.std(self.max_gain_budget))
        print('Avg percentage episode gain:', 1 - self.trial_num * sum(episode_gains) / sum(self.S).item())  #np.mean(episode_gain_ratios))
        print('Avg percentage max gain:', 1 - self.trial_num * sum(episode_max_gains) / sum(self.S).item())  #np.mean(episode_max_gain_ratios))
        print('Percentage of instances with positive gain:', len([x for x in episode_max_gains if x > 0]) / self.batch_size)

# if __name__ == '__main__':
#
#     folder = '/u/fy4bc/code/research/RL4CombOptm/Models/dqn_0114_base/'
#     with open(folder + 'dqn_' + str(5000), 'rb') as model_file:
#         model = pickle.load(model_file)
#
#     with open('/u/fy4bc/code/research/RL4CombOptm/Data/qiter33/qtable_chunk_18_0', 'rb') as data_file:
#         data_test = pickle.load(data_file)
#
#     # problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
#
#     problem = [data_test[i][0] for i in range(100)]
#     for i in range(100):
#         problem[i].g.ndata['h'] = torch.zeros((9, 16)).cuda()
#         problem[i].reset_label(label=[0,0,0,1,1,1,2,2,2], calc_S=False, rewire_edges=True)
#
#     test1 = test_summary(alg=model, problem=problem[0:1], num_instance=1)
#     test1.run_test(episode_len=50, explore_prob=0, time_aware=False)
#     test1.show_result()
#     print(test1.episodes[0].reward_seq)
#     print(test1.episodes[0].action_seq)
#
#     g0 = dc(test1.episodes[0].init_state)
#     g0.ndata['x'] = torch.tensor([[10.4788, 10.0846],
#             [10.9617, 10.5331],
#             [10.9883, 10.0443],
#             [10.0132, 10.4468],
#             [10.4449, 10.2032],
#             [10.7437, 10.8812],
#             [10.1346, 10.3417],
#             [10.5789, 10.0165],
#             [10.6024, 10.0736]], device='cuda:0')
#     p0 = dc(problem[0])
#     p0.reset_label(label=[0,0,0,1,1,1,2,2,2], calc_S=False, rewire_edges=True)
#     legal_actions = p0.get_legal_actions()
#     print(legal_actions)
#     S_a_encoding, h1, h2, Q_sa = model.model(dgl.batch([g0]), legal_actions.cuda(), gnn_step=3)
#     print(g0.ndata['label'])
#     print(g0.ndata['x'])
#     print(g0.edata['e_type'].view(9,-1))
#     print(h1)
#
#     print(legal_actions[Q_sa.argmax()])
#     g1, r1 = p0.step(legal_actions[Q_sa.argmax()])
#     print(r1)
#     g1 = to_cuda(g1)
#
#     legal_actions = p0.get_legal_actions()
#     print(legal_actions)
#     S_a_encoding, h1, h2, Q_sa = model.model(dgl.batch([g1]), legal_actions.cuda(), gnn_step=3)
#     print(Q_sa)
#     print(legal_actions[Q_sa.argmax()])
#     g2, r2 = p0.step(legal_actions[Q_sa.argmax()])
#     print(r2)
#
#     test_problem = test.episodes[0].
#     S = test_problem.calc_S()
#     g = to_cuda(test_problem.g)
#     ep = EpisodeHistory(g, max_episode_len=episode_len)
#
#     i = 0
#     legal_actions = test_problem.get_legal_actions()
#
#     S_a_encoding, h1, h2, Q_sa = alg.model(dgl.batch([g]), legal_actions.cuda(), gnn_step=3)
#
#     # epsilon greedy strategy
#     if torch.rand(1) > explore_prob:
#         action_idx = Q_sa.argmax()
#     else:
#         action_idx = torch.randint(high=legal_actions.shape[0], size=(1,)).squeeze()
#     swap_i, swap_j = legal_actions[action_idx]
#     state, reward = test_problem.step((swap_i, swap_j))
#     ep.write((swap_i, swap_j), action_idx, reward)
#     g = to_cuda(state)
#
#
#     # buf = alg.experience_replay_buffer[-1]
#     #
#     # g1 = problem.g
#     # g1a = problem.get_legal_actions()
#     # problem.reset()
#     # g2 = problem.g
#     # g2a = problem.get_legal_actions()
#     # bg = dgl.batch([g1, g2])
#     # model = model.cpu()
#     # sa1, _, h1, q1 = model(g1, g1a)
#     # sa2, _, h2, q2 = model(g2, g2a)
#     #
#     # ga = torch.cat([g1a, g2a], axis=0)
#     #
#     # torch.norm(torch.cat([sa1, sa2], axis=0) - sa)
#     # torch.norm(torch.cat([h1, h2], axis=0) - h)
#     # torch.norm(torch.cat([q1, q2], axis=0) - q)
#     # sa, _, h, q = model(bg, ga)
#     #
#     # a = conv(g1, aa)
#     # b = conv(g2, bb)
#     # c = conv(bg, torch.cat([aa, bb], axis=0))
#
#     # baseline = test_summary(alg=alg, problem=problem, num_instance=100)
#     # baseline.run_test(explore_prob=1.0)
#     # baseline.show_result()
#     folder = '/u/fy4bc/code/research/RL4CombOptm' + '/Models/dqn_0114_base/'
#     # folder = 'Models/dqn_0116_nox/'
#     # folder = 'Models/dqn_test_not_sample_batch_episode/'
#     with open(folder + 'dqn_' + str(5000), 'rb') as model_file:
#         model = pickle.load(model_file)
#     problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
#     test1 = test_summary(alg=model, problem=problem, num_instance=100)
#     test1.run_test(episode_len=50, explore_prob=0, time_aware=False)
#     test1.show_result()
#
#     test2 = test_summary(alg=alg, problem=problem, num_instance=100)
#     test2.run_test(episode_len=50, explore_prob=0, time_aware=False)
#     test2.show_result()
#
#
#     problem.calc_S(alg.experience_replay_buffer[1].init_state)
#     y = 0
#     z = 0
#     for i in range(test1.S.__len__()):
#         s = problem.calc_S(alg.experience_replay_buffer[i].init_state)
#         traj = 1 - np.cumsum(alg.experience_replay_buffer[i].reward_seq) / s
#         y += torch.cat([torch.tensor([1.]), traj], axis=0)
#     for i in range(test2.S.__len__()):
#         s = test2.S[i]
#         traj = 1 - np.cumsum(test2.episodes[i].reward_seq) / s
#         a = [1]
#         a.extend(list(traj))
#         z += np.array(a)
#     plt.figure(figsize=[10, 5])
#     plt.subplot(121)
#     plt.plot(y/100)
#     plt.xlabel('action steps')
#     plt.ylabel('avg. gain in S(%)')
#     plt.title('training')
#     plt.subplot(122)
#     plt.plot(z/100)
#     plt.xlabel('action steps')
#     plt.ylabel('avg. gain in S(%)')
#     plt.title('testing')
#     plt.savefig('Analysis/' + 'trajectory_vis_2' + '.png')
#     plt.close()
#
#
#
#     x = []
#     for i in range(100):
#         buf = alg.experience_replay_buffer[i]
#         best_s = max(np.cumsum(buf.reward_seq))
#         x.append(sum(buf.reward_seq))
#     buf.action_seq
#     buf.action_indices = [tensor(14), tensor(15), tensor(0), tensor(1), tensor(13)]
#     buf.reward_seq = tensor([ 0.4035, -0.4050,  0.5623, -0.2317,  0.3374])
#
#     alg.experience_replay_buffer[0].action_seq
#
#     np.cumsum(alg.experience_replay_buffer[0].reward_seq)
#
#
#
#
#     x = []
#     for i in tqdm(range(100)):
#         buf = alg_q_110.experience_replay_buffer[i]
#         x.append(sum(buf.reward_seq))
#     np.argmax(x)
#
#     buf = alg_q_110.experience_replay_buffer[39]
#     buf.action_seq
#     buf.reward_seq
#
#     alg.run_episode(gnn_step=3)
#
#     buf = alg.experience_replay_buffer[0]
#     buf.action_seq
#     buf.reward_seq
#
#     [(tensor(6, device='cuda:0'), tensor(7, device='cuda:0')),
#      (tensor(0, device='cuda:0'), tensor(6, device='cuda:0')),
#      (tensor(4, device='cuda:0'), tensor(5, device='cuda:0')),
#      (tensor(4, device='cuda:0'), tensor(7, device='cuda:0')),
#      (tensor(5, device='cuda:0'), tensor(8, device='cuda:0')),
#      (tensor(1), tensor(8)),
#      (tensor(0), tensor(1)),
#
#     problem = KCut_DGL(k=3, m=3, adjacent_reserve=5, hidden_dim=16)
#     buf = alg.experience_replay_buffer[0]
#     buf.init_state.ndata['label']
#     problem.g = dc(buf.init_state)
#
#     l = buf.init_state.ndata['label'][[8, 4, 3, 2, 6, 7, 1, 5, 0],:].nonzero()[:,1]
#
#     problem.reset_label([0, 1, 0, 2, 2, 1, 2, 1, 0])
#     problem.calc_S()
#     problem.step((1,8))
#     vis_g(problem, name='toy_models/0109_4', topo='c')
#
#
#
#     sa_,_,_,Q_sa_=alg.model(to_cuda(problem.g), problem.get_legal_actions())
#     [-0.9731,  0.4071, -0.2058,  0.1090,  0.7073, -0.7521,  0.7732, -1.1587,
#              0.1465,  1.4894, -0.4074, -1.0282,  0.4630, -0.3669, -0.1107, -0.0029,
#              0.0052,  0.4920, -0.1909,  0.0461,  0.9827,  0.3718, -0.4183, -0.2039,
#              0.2517, -0.3922,  0.0429,  0.0043, -0.0671, -0.3431,  0.3735, -0.0446,
#             -0.9731,  0.0697,  0.4810,  0.1089, -0.0939,  0.1573, -0.2550,  0.4802,
#             -0.2886, -0.2930,  0.0931,  0.5797, -0.4734, -0.3426, -0.1073,  0.4440,
#             -0.1909, -0.3064]
#     problem.step((4,8))
#     _,_,_,Q_sa1=alg.model(to_cuda(problem.g), problem.get_legal_actions())
#     [ 2.6993e-04,  1.5106e-01,  2.0658e-01,  2.2934e-02,  1.8874e-01,
#              1.8800e-01, -7.8708e-03,  1.6216e-01,  2.1767e-01,  1.4793e-02,
#              1.9973e-01,  1.9841e-01,  1.6251e-01,  1.2036e-01,  1.2011e-01,
#              1.5385e-01,  2.1907e-01,  1.3744e-01,  1.3767e-01,  1.7969e-01,
#              1.4820e-01,  *3.0644e-01*,  1.3666e-01,  1.3641e-01,  1.7007e-01,
#              2.3731e-01,  2.3643e-01]
#
#     [1,2,2,0,0,1,1,0,2]
#     problem.calc_S() 4.1640
#     [2, 6], [0, 3], [0, 3], [4, 6], [1, 6]
#     [ 0.9560, -0.6474,  0.6474, -0.3616,  0.0909, -0.9082,  0.9082, -0.9082,
#              0.9082,  0.6987, -1.0706,  0.3291,  0.9592, -1.1792,  0.3503, -0.3503,
#              0.3503, -0.2162,  0.1179,  0.0304]
#     problem.reset_label([1,2,2,0,0,1,1,0,2])
#     problem.step((0,3))
#     buf.action_indices
#     problem.calc_S()
#
#     x = []
#     for i in range(100):
#         x.append(alg.experience_replay_buffer[i].reward_seq[0])
