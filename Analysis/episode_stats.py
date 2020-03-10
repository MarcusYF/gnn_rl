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

    def test_dqn(self, alg, g_i, t, path=None):

        init_state = dc(self.episodes[g_i].init_state)
        label_history = init_state.ndata['label'][self.episodes[g_i].label_perm]

        self.problem.g = dc(init_state)
        self.problem.reset_label(label=label_history[0].argmax(dim=1))
        self.problem.calc_S()
        if path is not None:
            path = os.path.abspath(os.path.join(os.getcwd())) + path
            vis_g(self.problem, name=path + str(0), topo='cut')

        for i in range(t):
            actions = self.problem.get_legal_actions(state=self.problem.g)
            S_a_encoding, h1, h2, Q_sa = alg.forward(to_cuda(self.problem.g), actions.cuda(), gnn_step=3)
            _, r = self.problem.step(state=self.problem.g, action=actions[torch.argmax(Q_sa)])

            print(Q_sa.detach().cpu().numpy())
            print('action index:', torch.argmax(Q_sa).detach().cpu().item())
            print('action:', actions[torch.argmax(Q_sa)].detach().cpu().numpy())
            print('reward:', r.item())
            if path is not None:
                vis_g(self.problem, name=path + str(i+1), topo='cut')

    def compare_result(self):

        for gi in range(self.batch_size):

            print('Analyze', gi)
            opt, rs = self.cmpt_optimal(self.episodes[gi].init_state)
            grd, R, rs = self.test_greedy(self.episodes[gi].init_state)
            self.test_dqn(self.alg, gi, 7)

    def run_test(self, problem=None, trial_num=1, batch_size=100, gnn_step=3, episode_len=50, explore_prob=0.1, Temperature=1.0):
        self.num_actions = self.problem.N * (self.problem.N - self.problem.m) // 2 + 1
        self.trial_num = trial_num
        self.batch_size = batch_size
        if problem is None:
            batch_size *= trial_num
            bg = self.problem.gen_batch_graph(batch_size=self.batch_size)
            self.bg = to_cuda(self.problem.gen_batch_graph(x=bg.ndata['x'].repeat(trial_num, 1), batch_size=batch_size))
            gl = dgl.unbatch(self.bg)
            test_problem = self.problem
        else:
            # for validation problem set
            self.bg = problem
            gl = dgl.unbatch(self.bg)
            self.problem.g = self.bg
            test_problem = self.problem

        self.action_mask = torch.tensor(range(0, self.num_actions * batch_size, self.num_actions)).cuda()

        self.S = [test_problem.calc_S(g=gl[i]).cpu() for i in range(batch_size)]

        ep = [EpisodeHistory(gl[i], max_episode_len=episode_len, action_type='swap') for i in range(batch_size)]
        self.end_of_episode = {}.fromkeys(range(batch_size), episode_len-1)
        for i in tqdm(range(episode_len)):
            batch_legal_actions = test_problem.get_legal_actions(state=self.bg, action_type='swap').cuda()

            # if self.forbid_revisit and i > 0:
            #
            #     previous_actions = torch.tensor([ep[k].action_seq[i-1][0] * (self.n ** 2) + ep[k].action_seq[i-1][1] for k in range(batch_size)])
            #     bla = (self.n ** 2) * batch_legal_actions.view(batch_size, -1, 2)[:, :, 0] + batch_legal_actions.view(batch_size, -1, 2)[:, :, 1]
            #     forbid_action_mask = (bla.t() == previous_actions.cuda())
            #     batch_legal_actions = batch_legal_actions * (1 - forbid_action_mask.int()).t().flatten().unsqueeze(1)

            forbid_action_mask = torch.zeros(batch_legal_actions.shape[0], 1).cuda()
            if self.forbid_revisit:
                for k in range(batch_size):
                    cur_state = ep[k].enc_state_seq[-1]  # current state for the k-th episode
                    forbid_states = set(ep[k].enc_state_seq[-1-self.forbid_revisit:-1])
                    candicate_actions = batch_legal_actions[k*self.num_actions:(k+1)*self.num_actions,:]
                    for j in range(self.num_actions):
                        cur_state_l = QtableKey2state(cur_state)
                        cur_state_l[candicate_actions[j][0]] ^= cur_state_l[candicate_actions[j][1]]
                        cur_state_l[candicate_actions[j][1]] ^= cur_state_l[candicate_actions[j][0]]
                        cur_state_l[candicate_actions[j][0]] ^= cur_state_l[candicate_actions[j][1]]
                        if state2QtableKey(cur_state_l) in forbid_states:
                            forbid_action_mask[j + k * self.num_actions] += 1

            batch_legal_actions = batch_legal_actions * (1 - forbid_action_mask.int()).t().flatten().unsqueeze(1)

            if self.q_net == 'mlp':
                S_a_encoding, h1, h2, Q_sa = self.alg.forward(self.bg, batch_legal_actions, gnn_step=gnn_step)
            else:
                S_a_encoding, h1, h2, Q_sa = self.alg.forward_MHA(self.bg, batch_legal_actions, gnn_step=gnn_step)

            terminate_episode = (Q_sa.view(-1, self.num_actions).max(dim=1).values < 0.).nonzero().flatten().cpu().numpy()

            for idx in terminate_episode:
                if self.end_of_episode[idx] == episode_len - 1:
                    self.end_of_episode[idx] = i

            best_actions = torch.multinomial(F.softmax(Q_sa.view(-1, self.num_actions) / Temperature), 1).view(-1)

            chose_actions = torch.tensor(
                [x if torch.rand(1) > explore_prob else torch.randint(high=self.num_actions, size=(1,)).squeeze() for x in
                 best_actions]).cuda()
            chose_actions += self.action_mask

            actions = batch_legal_actions[chose_actions]

            _, rewards, sub_rewards = self.problem.step_batch(states=self.bg, action=actions, return_sub_reward=True)

            R = [reward.item() for reward in rewards]

            enc_states = [state2QtableKey(self.bg.ndata['label'][k * self.n: (k + 1) * self.n].argmax(dim=1).cpu().numpy()) for k in range(batch_size)]

            [ep[k].write(action=actions[k, :], action_idx=best_actions[k] - self.action_mask[k], reward=R[k]
                         , q_val=Q_sa.view(-1, self.num_actions)[k, :]
                         , actions=batch_legal_actions.view(-1, self.num_actions, 2)[k, :, :]
                         , state_enc=enc_states[k]
                         , sub_reward=sub_rewards[:, k].cpu().numpy()) for k in range(batch_size)]

        self.episodes = ep

    def compare_opt(self, validation_problem):

        bingo = 0
        for i in range(self.batch_size):
            self.problem.g = self.episodes[i].init_state
            end_perm = self.episodes[i].label_perm[self.end_of_episode[i]]
            end_label = self.episodes[i].init_state.ndata['label'][end_perm]
            dqn_result = state2QtableKey(end_label.argmax(dim=1).cpu().numpy())
            opt_result = state2QtableKey(validation_problem[i][1])
            grd_result = state2QtableKey(validation_problem[i][2])
            if dqn_result == opt_result:
                bingo += 1
        print('Optimal solution hit percentage:', bingo)
        return bingo

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
        episode_max_gain_steps = []
        episode_max_gain_ratios = []
        for i in select_indices:
            initial_S.append(self.S[i].item())
            episode_gains.append(self.S[i].item() - sum(self.episodes[i].reward_seq[:self.end_of_episode[i]]))
            episode_max_gains.append(self.S[i].item() - max(max(np.cumsum(self.episodes[i].reward_seq)), 0))
            episode_max_gain_steps.append(np.argmax(np.cumsum(self.episodes[i].reward_seq)) + 1)
            episode_gain_ratios.append(episode_gains[-1] / self.S[i].item())
            episode_max_gain_ratios.append(max(episode_max_gains[-1], 0) / self.S[i].item())

            # if episode_max_gains[-1] < episode_gains[-1]:
            #     print(i)
            #     print(self.episodes[i].reward_seq)
        print('Avg value of initial S:', np.mean(self.S))
        print('Avg episode end value:', np.mean(episode_gains))
        print('Avg episode best value:', np.mean(episode_max_gains))
        print('Avg episode step budget(Avg/Max/Min):', np.mean(episode_max_gain_steps), np.max(episode_max_gain_steps), np.min(episode_max_gain_steps))
        # print('Avg max gain budget:', np.mean(self.max_gain_budget))
        # print('Var max gain budget:', np.std(self.max_gain_budget))
        print('Avg percentage episode gain:', 1 - self.trial_num * sum(episode_gains) / sum(self.S).item())  #np.mean(episode_gain_ratios))
        print('Avg percentage max gain:', 1 - self.trial_num * sum(episode_max_gains) / sum(self.S).item())  #np.mean(episode_max_gain_ratios))
        print('Percentage of instances with positive gain:', len([x for x in episode_max_gains if x > 0]) / self.batch_size)
        return np.mean(self.S) - np.mean(episode_gains)
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
