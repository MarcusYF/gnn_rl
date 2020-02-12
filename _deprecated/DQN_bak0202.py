# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:20:39 2018
@author: fy4bc
"""
import time
import numpy as np
from dgl.graph import DGLGraph
from log_utils import mean_val, logger
from k_cut import *
from dataclasses import dataclass
import pickle


def to_cuda(G_):
    G = dc(G_)
    G.ndata['x'] = G.ndata['x'].cuda()
    G.ndata['label'] = G.ndata['label'].cuda()
    G.ndata['h'] = G.ndata['h'].cuda()
    G.edata['d'] = G.edata['d'].cuda()
    G.edata['w'] = G.edata['w'].cuda()
    G.edata['e_type'] = G.edata['e_type'].cuda()
    return G


class EpisodeHistory:
    def __init__(self, g, max_episode_len, action_type='swap'):
        self.action_type = action_type
        self.init_state = dc(g)

        self.n = g.number_of_nodes()
        self.max_episode_len = max_episode_len
        self.episode_len = 0
        self.action_seq = []
        self.action_indices = []
        self.reward_seq = []
        self.empl_reward_seq = []
        # self.calib_reward_seq = []
        if self.action_type == 'swap':
            self.label_perm = torch.tensor(range(self.n)).unsqueeze(0)
        if self.action_type == 'flip':
            self.label_perm = self.init_state.ndata['label'].nonzero()[:, 1].unsqueeze(0)
        # self.visited_state = set([''.join([str(i.item()) for i in torch.tensor(range(self.n)).unsqueeze(0)[0]])])
        # self.node_visit_cnt = [0] * self.n
        self.best_gain_sofar = 0
        self.current_gain = 0

    def perm_label(self, label, action):
        label = dc(label)
        if self.action_type == 'swap':
            tmp = dc(label[action[0]])
            label[action[0]] = label[action[1]]
            label[action[1]] = tmp
        if self.action_type == 'flip':
            label[action[0]] = action[1]
        return label.unsqueeze(0)

    # def check_loop(self, action):
    #
    #     new_label = self.perm_label(self.label_perm[-1, :], action)
    #     new_label_str = ''.join([str(i.item()) for i in new_label[0]])
    #     if new_label_str in self.visited_state:
    #         return True
    #     else:
    #         return False

    def write(self, action, action_idx, reward):


        new_label = self.perm_label(self.label_perm[-1, :], action)

        # new_label_str = ''.join([str(i.item()) for i in new_label[0]])
        # self.visited_state.add(new_label_str)
        self.action_seq.append(action)

        self.action_indices.append(action_idx)

        self.reward_seq.append(reward)

        self.empl_reward_seq.append(reward * 10 - 1)

        # self.current_gain += reward

        # self.calib_reward_seq.append(max(self.best_gain_sofar, self.current_gain) - self.best_gain_sofar)

        # print('best gain previous:', self.best_gain_sofar)
        # print('current gain:', self.current_gain)
        # self.best_gain_sofar = max(self.best_gain_sofar, self.current_gain)
        # print('renewed best gain:', self.best_gain_sofar)
        # print(reward, self.current_gain, self.best_gain_sofar)

        self.label_perm = torch.cat([self.label_perm, new_label], dim=0)
        # self.node_visit_cnt[action[0]] += 1
        # self.node_visit_cnt[action[1]] += 1
        self.episode_len += 1

    def wrap(self):
        self.reward_seq = torch.tensor(self.reward_seq)
        self.empl_reward_seq = torch.tensor(self.empl_reward_seq)
        # self.calib_reward_seq = torch.tensor(self.calib_reward_seq)
        self.label_perm = self.label_perm.long()



@dataclass
class sars:
    s0: DGLGraph
    a: tuple
    r: float
    s1: DGLGraph

def weight_monitor(model, model_target):
    # gnn_0 = torch.mean(model.layers[0].apply_mod.l0.weight).item(), torch.std(model.layers[0].apply_mod.l0.weight).item()
    # gnn_1 = torch.mean(model.layers[0].apply_mod.l1.weight).item(), torch.std(model.layers[0].apply_mod.l1.weight).item()
    # gnn_2 = torch.mean(model.layers[0].apply_mod.l2.weight).item(), torch.std(model.layers[0].apply_mod.l2.weight).item()
    # gnn_3 = torch.mean(model.layers[0].apply_mod.l3.weight).item(), torch.std(model.layers[0].apply_mod.l3.weight).item()
    # gnn_4 = torch.mean(model.layers[0].apply_mod.l4.weight).item(), torch.std(model.layers[0].apply_mod.l4.weight).item()
    # gnn_5 = torch.mean(model.layers[0].apply_mod.l5.weight).item(), torch.std(model.layers[0].apply_mod.l5.weight).item()
    # attn = [(torch.mean(model.MHA.linears[i].weight).item(), torch.std(model.MHA.linears[i].weight).item()) for i in range(4)]
    # q_net_1 = torch.mean(model.value1.weight).item(), torch.std(model.value1.weight).item()
    # q_net_2 = torch.mean(model.value2.weight).item(), torch.std(model.value2.weight).item()
    # gnn_diff0 = torch.norm(model.layers[0].apply_mod.l0.weight - model_target.layers[0].apply_mod.l0.weight).item()
    # gnn_diff1 = torch.norm(model.layers[0].apply_mod.l1.weight - model_target.layers[0].apply_mod.l1.weight).item()
    # gnn_diff2 = torch.norm(model.layers[0].apply_mod.l2.weight - model_target.layers[0].apply_mod.l2.weight).item()
    # gnn_diff3 = torch.norm(model.layers[0].apply_mod.l3.weight - model_target.layers[0].apply_mod.l3.weight).item()
    # gnn_diff4 = torch.norm(model.layers[0].apply_mod.l4.weight - model_target.layers[0].apply_mod.l4.weight).item()
    # gnn_diff5 = torch.norm(model.layers[0].apply_mod.l5.weight - model_target.layers[0].apply_mod.l5.weight).item()
    # attn_diff0 = torch.norm(model.MHA.linears[0].weight - model_target.MHA.linears[0].weight).item()
    # attn_diff1 = torch.norm(model.MHA.linears[1].weight - model_target.MHA.linears[1].weight).item()
    # attn_diff2 = torch.norm(model.MHA.linears[2].weight - model_target.MHA.linears[2].weight).item()
    # attn_diff3 = torch.norm(model.MHA.linears[3].weight - model_target.MHA.linears[3].weight).item()
    # q_net_diff1 = torch.norm(model.value1.weight - model_target.value1.weight).item()
    # q_net_diff2 = torch.norm(model.value2.weight - model_target.value2.weight).item()
    # return {'gnn0':gnn_0, 'gnn1':gnn_1, 'gnn2':gnn_2, 'gnn3':gnn_3, 'gnn4':gnn_4, 'gnn5':gnn_5,
    #         'attn':attn, 'q_net1':q_net_1, 'q_net2':q_net_2, 'gnn_diff0':gnn_diff0,
    #         'gnn_diff1':gnn_diff1, 'gnn_diff2':gnn_diff2, 'gnn_diff3':gnn_diff3, 'gnn_diff4':gnn_diff4, 'gnn_diff5':gnn_diff5,
    #         'attn_diff0':attn_diff0, 'attn_diff1':attn_diff1, 'attn_diff2':attn_diff2, 'attn_diff3':attn_diff3,
    #         'q_net_diff1':q_net_diff1, 'q_net_diff2':q_net_diff2}

    gnn_0 = torch.mean(model.layers[0].apply_mod.l0.weight).item(), torch.std(model.layers[0].apply_mod.l0.weight).item()
    gnn_1 = torch.mean(model.layers[0].apply_mod.l1.weight).item(), torch.std(model.layers[0].apply_mod.l1.weight).item()
    gnn_2 = torch.mean(model.layers[0].apply_mod.l2.weight).item(), torch.std(model.layers[0].apply_mod.l2.weight).item()
    gnn_3 = torch.mean(model.layers[0].apply_mod.t3.weight).item(), torch.std(model.layers[0].apply_mod.l3.weight).item()
    gnn_4 = torch.mean(model.layers[0].apply_mod.t4.weight).item(), torch.std(model.layers[0].apply_mod.l4.weight).item()


    gnn_diff0 = torch.norm(model.layers[0].apply_mod.l0.weight - model_target.layers[0].apply_mod.l0.weight).item()
    gnn_diff1 = torch.norm(model.layers[0].apply_mod.l1.weight - model_target.layers[0].apply_mod.l1.weight).item()
    gnn_diff2 = torch.norm(model.layers[0].apply_mod.l2.weight - model_target.layers[0].apply_mod.l2.weight).item()
    gnn_diff3 = torch.norm(model.layers[0].apply_mod.t3.weight - model_target.layers[0].apply_mod.t3.weight).item()
    gnn_diff4 = torch.norm(model.layers[0].apply_mod.t4.weight - model_target.layers[0].apply_mod.t4.weight).item()

    q_net_1 = torch.mean(model.t5.weight).item(), torch.std(model.t5.weight).item()
    q_net_2 = torch.mean(model.t6.weight).item(), torch.std(model.t6.weight).item()
    q_net_3 = torch.mean(model.t7.weight).item(), torch.std(model.t7.weight).item()

    q_net_diff1 = torch.norm(model.t5.weight - model_target.t5.weight).item()
    q_net_diff2 = torch.norm(model.t6.weight - model_target.t6.weight).item()
    q_net_diff3 = torch.norm(model.t7.weight - model_target.t7.weight).item()


    return {'gnn0':gnn_0, 'gnn1':gnn_1, 'gnn2':gnn_2, 'gnn3':gnn_3, 'gnn4':gnn_4,
            'q_net1':q_net_1, 'q_net2':q_net_2, 'q_net3':q_net_3,
            'gnn_diff0':gnn_diff0, 'gnn_diff1':gnn_diff1, 'gnn_diff2':gnn_diff2, 'gnn_diff3':gnn_diff3, 'gnn_diff4':gnn_diff4,
            'q_net_diff1':q_net_diff1, 'q_net_diff2':q_net_diff2, 'q_net_diff3':q_net_diff3}


class DQN:
    def __init__(self, problem, action_type='swap', gamma=1.0, eps=0.1, lr=1e-4, replay_buffer_max_size=10, replay_buffer_max_size2=5000, extended_h=False, time_aware=False, use_x=False, edge_info='adj_weight', readout='mlp', clip_target=False, use_calib_reward=False, cuda_flag=True):

        self.problem = problem
        self.action_type = action_type
        self.G = problem.g  # the graph
        self.k = problem.k  # num of clusters
        self.m = problem.m  # num of nodes in cluster
        self.ajr = problem.adjacent_reserve  # degree of node in graph
        self.hidden_dim = problem.hidden_dim  # hidden dimension for node representation
        self.n = self.k * self.m  # num of nodes
        self.eps = eps  # constant for exploration in dqn
        self.extended_h = extended_h
        self.use_x = use_x
        self.edge_info = edge_info
        self.clip_target = clip_target
        self.use_calib_reward = use_calib_reward
        if cuda_flag:
            self.model = DQNet(k=self.k, m=self.m, ajr=self.ajr, num_head=4, hidden_dim=self.hidden_dim, extended_h=self.extended_h, use_x=self.use_x, edge_info=self.edge_info, readout=readout).cuda()
        else:
            self.model = DQNet(k=self.k, m=self.m, ajr=self.ajr, num_head=4, hidden_dim=self.hidden_dim, extended_h=self.extended_h, use_x=self.use_x, edge_info=self.edge_info, readout=readout)
        # self.model.apply(self.weights_init)  # initialize weight
        self.model_target = dc(self.model)
        self.gamma = gamma  # reward decay const
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.experience_replay_buffer = []
        self.experience_replay_buffer2 = []
        self.buffer_episode_offset = [0]
        self.buffer_indices = []
        self.replay_buffer_max_size = replay_buffer_max_size
        self.replay_buffer_max_size2 = replay_buffer_max_size2
        self.time_aware = time_aware
        self.cuda = cuda_flag
        self.log = logger()
        self.Q_err = 0  # Q error
        self.log.add_log('tot_return')
        self.log.add_log('Q_error')
        self.log.add_log('entropy')
        self.log.add_log('R_signal')

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            # init.xavier_uniform(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)
            # if m.bias is not None:
            #     torch.nn.init.xavier_uniform_(m.bias)

    def run_episode_(self, action_type='swap', gnn_step=3, episode_len=50, print_info=False):
        sum_r = 0
        state = self.problem.reset(compute_S=False)
        t = 0
        T1 = time.time()
        ep = EpisodeHistory(state, episode_len, action_type=action_type)
        T2 = time.time()
        # print('EpisodeHistory init:', T2-T1)
        terminal_flag = False

        while t < episode_len and not terminal_flag:

            legal_actions = self.problem.get_legal_actions(action_type=action_type)
            # epsilon greedy strategy
            if torch.rand(1) > self.eps:
                if self.cuda:
                    G = to_cuda(state)
                    legal_actions = legal_actions.cuda()
                else:
                    G = dc(state)

                S_a_encoding, h1, h2, Q_sa = self.model(dgl.batch([G]), legal_actions, action_type=action_type, gnn_step=gnn_step, time_aware=self.time_aware, remain_episode_len=episode_len-t-1)

                if print_info and (t % episode_len in (0, episode_len//2, episode_len-1)):
                    # print('step:', t)
                    # print(G.ndata['label'])
                    # print(Q_sa.argmax())
                    # record
                    h_support1 = h1.nonzero().shape[0]
                    h_support2 = h2.nonzero().shape[0]
                    h_mean = h1.sum() / h_support1
                    h_residual = self.model.h_residual
                    q_mean = Q_sa.mean()
                    q_var = Q_sa.std()
                    # model weight
                    print('\nh-nonzero entry: %.0f, %.0f'%(h_support1, h_support2))
                    # print('*****-------------------------------------*****\n')
                    # print(h2)
                    # print('*****-------------------------------------*****\n')
                    print('h-mean: %.5f'%h_mean.item())
                    # print('h-residual: ', ['%.2f'%x.item() for x in h_residual])
                    print('q value-mean: %.5f'%q_mean.item())
                    print('q value-std: %.5f'%q_var.item())
                    print(weight_monitor(self.model, self.model_target))

                best_action = Q_sa.argmax()
            else:
                best_action = torch.randint(high=legal_actions.shape[0], size=(1, )).squeeze()

            swap_i, swap_j = legal_actions[best_action]

            # # TODO: Re-design reward signal? What about the terminal state?
            # trial = 0
            # while ep.check_loop(action=(swap_i, swap_j)) and trial < 10:
            #     random_action = torch.randint(high=legal_actions.shape[0], size=(1, )).squeeze()
            #     swap_i, swap_j = legal_actions[random_action]
            #     trial += 1
            T1 = time.time()
            state, reward = self.problem.step((swap_i, swap_j), action_type=action_type)
            T2 = time.time()

            ep.write(action=(swap_i, swap_j), action_idx=best_action, reward=reward.item())

            sum_r += reward

            # print('step time:', T2 - T1)

            if t == 0:
                R = reward.unsqueeze(0)
            else:
                R = torch.cat([R, reward.unsqueeze(0)], dim=0)

            # terminal_flag = max(ep.node_visit_cnt) > 5
            t += 1

        ep.wrap()

        self.experience_replay_buffer.append(ep)
        self.buffer_episode_offset.append(self.buffer_episode_offset[-1] + ep.episode_len)
        self.buffer_indices.append(list(range(ep.episode_len)))

        self.log.add_item('tot_return', sum_r.item())
        tot_return = R.sum().item()

        return R, tot_return

    def run_batch_episode_(self, action_type='swap', gnn_step=3, episode_len=50, batch_size=10, print_info=False):

        sum_r = 0
        state = self.problem.gen_batch_graph(batch_size=batch_size)
        t = 0

        ep = [EpisodeHistory(state[i], episode_len, action_type=action_type) for i in range(batch_size)]

        num_actions = self.problem.get_legal_actions(action_type=action_type).shape[0]
        action_mask = torch.tensor(range(0, num_actions * batch_size, num_actions)).cuda()

        while t < episode_len:

            legal_actions = [self.problem.get_legal_actions(state=state[i], action_type=action_type).cuda() for i in range(batch_size)]

            batch_legal_actions = torch.cat(legal_actions, axis=0)

            # epsilon greedy strategy
            if torch.rand(1) > self.eps:

                G = [to_cuda(state[i]) for i in range(batch_size)]

                batch_G = dgl.batch(G)

                S_a_encoding, h1, h2, Q_sa = self.model(batch_G, batch_legal_actions, action_type=action_type, gnn_step=gnn_step, time_aware=self.time_aware, remain_episode_len=episode_len-t-1)

                if print_info and (t % episode_len in (0, episode_len//2, episode_len-1)):
                    h_support1 = h1.nonzero().shape[0]
                    h_support2 = h2.nonzero().shape[0]
                    h_mean = h1.sum() / h_support1
                    q_mean = Q_sa.mean()
                    q_var = Q_sa.std()
                    # model weight
                    print('\nh-nonzero entry: %.0f, %.0f'%(h_support1, h_support2))
                    print('h-mean: %.5f'%h_mean.item())
                    print('q value-mean: %.5f'%q_mean.item())
                    print('q value-std: %.5f'%q_var.item())
                    print(weight_monitor(self.model, self.model_target))

                best_actions = action_mask + Q_sa.view(-1, num_actions).argmax(dim=1)
            else:
                best_actions = action_mask + torch.randint(high=num_actions, size=(batch_size, )).squeeze().cuda()

            swap_ij = batch_legal_actions[best_actions]

            # TODO: Re-design reward signal? What about the terminal state?
            state_reward_tuples = [self.problem.step(action=swap_ij[k, :], action_type=action_type, state=state[k]) for k in range(batch_size)]

            R = [state_reward[1].unsqueeze(0) for state_reward in state_reward_tuples]
            sum_r += sum(R)

            # print(R[0].squeeze(0).item())
            [ep[k].write(action=swap_ij[k, :], action_idx=best_actions[k]-action_mask[k], reward=R[k].item()) for k in range(batch_size)]

            t += 1

        [e.wrap() for e in ep]

        self.experience_replay_buffer.extend(ep)
        self.buffer_episode_offset.extend([self.buffer_episode_offset[-1] + ep[0].episode_len * k for k in range(batch_size)])
        self.buffer_indices.extend([list(range(ep[0].episode_len))] * batch_size)
        self.log.add_item('tot_return', sum_r.item())
        tot_return = sum(R).item()

        return R, tot_return

    def run_batch_episode(self, action_type='swap', gnn_step=3, episode_len=50, batch_size=10, print_info=False):

        sum_r = 0
        bg = to_cuda(self.problem.gen_batch_graph(batch_size=batch_size))
        t = 0

        num_actions = self.problem.get_legal_actions(action_type=action_type).shape[0]
        action_mask = torch.tensor(range(0, num_actions * batch_size, num_actions)).cuda()

        explore_dice = (torch.rand(episode_len, batch_size) < self.eps)
        explore_replace_mask = explore_dice.nonzero()  #
        explore_step_offset = torch.cat([torch.zeros([1], dtype=torch.long), torch.cumsum(explore_dice.sum(dim=1), dim=0)], dim=0)
        explore_replace_actions = torch.randint(high=num_actions, size=(explore_replace_mask.shape[0], )).cuda()

        while t < episode_len:

            batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=action_type).cuda()

            # epsilon greedy strategy
            #TODO: cannot dc bg once being forwarded.
            _, _, _, Q_sa = self.model(dc(bg), batch_legal_actions, action_type=action_type, gnn_step=gnn_step, time_aware=self.time_aware, remain_episode_len=episode_len-t-1)
            best_actions = Q_sa.view(-1, num_actions).argmax(dim=1)

            # chosen_actions = torch.tensor([x if torch.rand(1) > self.eps else torch.randint(high=num_actions, size=(1,)).squeeze() for x in best_actions]).cuda()
            # chosen_actions += action_mask
            # actions = batch_legal_actions[chosen_actions]
            explore_episode_indices = explore_replace_mask[explore_step_offset[t]: explore_step_offset[t + 1]][:, 1]
            explore_actions = explore_replace_actions[explore_step_offset[t]: explore_step_offset[t + 1]]
            best_actions[explore_episode_indices] = explore_actions
            best_actions += action_mask

            actions = batch_legal_actions[best_actions]

            # update bg inplace and calculate batch rewards
            g0 = [g for g in dgl.unbatch(dc(bg))]  # current_state
            _, rewards = self.problem.step_batch(states=bg, action=actions)
            g1 = [g for g in dgl.unbatch(dc(bg))]  # after_state

            self.experience_replay_buffer2.extend([sars(g0[i], actions[i], rewards[i], g1[i]) for i in range(batch_size)])

            R = [reward.item() for reward in rewards]
            sum_r += sum(R)

            t += 1

        self.log.add_item('tot_return', sum_r)

        return R

    def sample_from_buffer(self, epoch, batch_size, q_step, gnn_step, episode_len, ddqn):

        flat_ep_i = [(x[0], y) for x in zip(range(len(self.buffer_indices)), self.buffer_indices) for y in x[1][:-q_step]]
        # sample #batch_size indices in replay buffer
        batch_size = min(batch_size, len(flat_ep_i))
        idx_start = random.sample(flat_ep_i, batch_size)

        # # locate samples in each episode
        # batch_idx = [(i // episode_len, i % episode_len) for i in idx]
        # locate start/end states for each sample
        # idx_start = [i for i in batch_idx if i[1] < self.experience_replay_buffer[i[0]].episode_len - q_step]

        R = []
        begin_state = []
        end_state = []
        begin_action = []
        end_action = []
        action_indices = []
        for episode_i, step_j in idx_start:

            action_idx = self.experience_replay_buffer[episode_i].action_indices[step_j]
            action_indices.append(action_idx)
            if self.use_calib_reward:
                # r = self.experience_replay_buffer[episode_i].calib_reward_seq[step_j: step_j + q_step]
                r = self.experience_replay_buffer[episode_i].empl_reward_seq[step_j: step_j + q_step]
            else:
                r = self.experience_replay_buffer[episode_i].reward_seq[step_j: step_j + q_step]
            r = torch.sum(r * torch.tensor([self.gamma ** i for i in range(q_step)]))
            R.append(r.unsqueeze(0))

            G_start = to_cuda(self.experience_replay_buffer[episode_i].init_state)
            G_end = to_cuda(self.experience_replay_buffer[episode_i].init_state)

            if self.action_type == 'swap':
                G_start.ndata['label'] = G_start.ndata['label'][self.experience_replay_buffer[episode_i].label_perm[step_j], :]
                G_end.ndata['label'] = G_end.ndata['label'][self.experience_replay_buffer[episode_i].label_perm[step_j + q_step], :]
            if self.action_type == 'flip':
                G_start.ndata['label'] = torch.nn.functional.one_hot(self.experience_replay_buffer[episode_i].label_perm[step_j], self.k).float()
                G_end.ndata['label'] = torch.nn.functional.one_hot(self.experience_replay_buffer[episode_i].label_perm[step_j + q_step], self.k).float()
            G_start.ndata['label'] = G_start.ndata['label'].cuda()
            G_end.ndata['label'] = G_end.ndata['label'].cuda()

            G_start_actions = self.problem.get_legal_actions(state=G_start, action_type=self.action_type)
            G_end_actions = self.problem.get_legal_actions(state=G_end, action_type=self.action_type)
            G_start_actions = G_start_actions.cuda()
            G_end_actions = G_end_actions.cuda()

            begin_state.append(G_start)
            end_state.append(G_end)
            begin_action.append(G_start_actions)
            end_action.append(G_end_actions)

        # make them batches
        batch_begin_state = dgl.batch(begin_state)
        batch_end_state = dgl.batch(end_state)
        batch_begin_action = torch.cat(begin_action, axis=0)
        batch_end_action = torch.cat(end_action, axis=0)

        ## old version
        # action_mask = torch.tensor(range(0, G_start_actions.shape[0] * batch_size, G_start_actions.shape[0])).cuda()
        # _, _, _, Q_s1a = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type,
        #                             gnn_step=gnn_step)
        # _, _, _, Q_s2a = self.model_target(batch_end_state, batch_end_action, action_type=self.action_type,
        #                                    gnn_step=gnn_step)
        #
        # q = self.gamma ** q_step * Q_s2a.view(-1, G_start_actions.shape[0]).max(dim=1).values
        # q -= Q_s1a[action_mask + torch.tensor(action_indices).cuda()]


        # estimate Q-values and calculate diff between Q-values at start/end
        action_mask = torch.tensor(range(0, G_start_actions.shape[0] * batch_size, G_start_actions.shape[0])).cuda()
        begin_action_list = action_mask + torch.tensor(action_indices).cuda()
        if not ddqn:
            # only compute limited number for Q_s1a
            _, _, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action[begin_action_list], action_type=self.action_type, gnn_step=gnn_step, time_aware=self.time_aware, remain_episode_len=episode_len-step_j-1)
            _, _, _, Q_s2a = self.model_target(batch_end_state, batch_end_action, action_type=self.action_type, gnn_step=gnn_step, time_aware=self.time_aware, remain_episode_len=episode_len-step_j-1-q_step)
            # Q_s2a = Q_s2a.detach()
            if self.clip_target:
                Q_s2a = F.relu(Q_s2a)
            if self.time_aware and step_j + q_step == episode_len - 1:
                q = - Q_s1a_
            else:
                q = self.gamma ** q_step * Q_s2a.view(-1, G_start_actions.shape[0]).max(dim=1).values - Q_s1a_
        else:
            _, _, _, Q_s1a = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step, time_aware=self.time_aware, remain_episode_len=episode_len-step_j-1)
            max_action_list = action_mask + Q_s1a.view(-1, G_start_actions.shape[0]).argmax(dim=1)
            _, _, _, Q_s2a_ = self.model_target(batch_end_state, batch_end_action[max_action_list], action_type=self.action_type, gnn_step=gnn_step, time_aware=self.time_aware, remain_episode_len=episode_len-step_j-1-q_step)
            if self.clip_target:
                Q_s2a_ = F.relu(Q_s2a_)
            if self.time_aware and step_j + q_step == episode_len - 1:
                q = - Q_s1a[begin_action_list]
            else:
                q = self.gamma ** q_step * Q_s2a_ - Q_s1a[begin_action_list]

        Q = q.unsqueeze(0)

        return torch.cat(R), Q


        # for episode_i, step_j in idx_start:
        #     # TODO: should run in parallel and avoid the for-loop (need to forward graph batches in dgl)
        #
        #     # calculate start/end states
        #     if self.cuda:
        #         G_start = to_cuda(self.experience_replay_buffer[episode_i].init_state)
        #         G_end = to_cuda(self.experience_replay_buffer[episode_i].init_state)
        #     else:
        #         G_start = dc(to_cuda(self.experience_replay_buffer[episode_i].init_state))
        #         G_end = dc(to_cuda(self.experience_replay_buffer[episode_i].init_state))
        #
        #     if self.action_type == 'swap':
        #         G_start.ndata['label'] = G_start.ndata['label'][self.experience_replay_buffer[episode_i].label_perm[step_j], :]
        #         G_end.ndata['label'] = G_end.ndata['label'][self.experience_replay_buffer[episode_i].label_perm[step_j + q_step], :]
        #     if self.action_type == 'flip':
        #         G_start.ndata['label'] = torch.nn.functional.one_hot(self.experience_replay_buffer[episode_i].label_perm[step_j], self.k).float()
        #         G_end.ndata['label'] = torch.nn.functional.one_hot(self.experience_replay_buffer[episode_i].label_perm[step_j + q_step], self.k).float()
        #
        #     G_start_actions = self.problem.get_legal_actions(state=G_start, action_type=self.action_type)
        #     G_end_actions = self.problem.get_legal_actions(state=G_end, action_type=self.action_type)
        #
        #     if self.cuda:
        #         G_start.ndata['label'] = G_start.ndata['label'].cuda()
        #         G_end.ndata['label'] = G_end.ndata['label'].cuda()
        #         G_start_actions = G_start_actions.cuda()
        #         G_end_actions = G_end_actions.cuda()
        #
        #     # estimate Q-values
        #     if self.time_aware:
        #         _, _, _, Q_s1a = self.model(G_start, G_start_actions, action_type=self.action_type, gnn_step=gnn_step, remain_episode_len=episode_len-step_j-1)
        #         _, _, _, Q_s2a = self.model_target(G_end, G_end_actions, action_type=self.action_type, gnn_step=gnn_step, remain_episode_len=episode_len-step_j-1-q_step)
        #     else:
        #         _, _, _, Q_s1a = self.model(G_start, G_start_actions, action_type=self.action_type, gnn_step=gnn_step)
        #         _, _, _, Q_s2a = self.model_target(G_end, G_end_actions, action_type=self.action_type, gnn_step=gnn_step)
        #
        #     # calculate accumulated reward
        #     swap_i, swap_j = self.experience_replay_buffer[episode_i].action_seq[step_j]
        #     action_idx = self.experience_replay_buffer[episode_i].action_indices[step_j]
        #     r = self.experience_replay_buffer[episode_i].reward_seq[step_j: step_j + q_step]
        #     r = torch.sum(r * torch.tensor([self.gamma ** i for i in range(q_step)]))
        #
        #     # calculate diff between Q-values at start/end
        #     if self.time_aware and step_j + q_step == episode_len - 1:
        #         q = 0
        #     else:
        #         if not ddqn:
        #             q = self.gamma ** q_step * Q_s2a.max()
        #         else:
        #             q = self.gamma ** q_step * Q_s2a[Q_s1a.argmax()]
        #     q -= Q_s1a[action_idx]
        #
        #     R.append(r.unsqueeze(0))
        #     Q.append(q.unsqueeze(0))
        #
        #     t += 1

        # return torch.cat(R), torch.cat(Q)

    def sample_from_buffer2(self, batch_size, q_step, gnn_step):

        batch_size = min(batch_size, len(self.experience_replay_buffer2))
        # sample_buffer = random.sample(self.experience_replay_buffer2, batch_size)
        sample_buffer = np.random.choice(self.experience_replay_buffer2, batch_size, replace=False)
        # make batches
        batch_begin_state = dgl.batch([tpl.s0 for tpl in sample_buffer])
        batch_end_state = dgl.batch([tpl.s1 for tpl in sample_buffer])
        R = [tpl.r.unsqueeze(0) for tpl in sample_buffer]
        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
        batch_end_action = self.problem.get_legal_actions(state=batch_end_state, action_type=self.action_type).cuda()
        action_num = batch_end_action.shape[0] // batch_begin_action.shape[0]

        # only compute limited number for Q_s1a
        _, _, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step)
        _, _, _, Q_s2a = self.model_target(batch_end_state, batch_end_action, action_type=self.action_type, gnn_step=gnn_step)
        # Q_s2a = Q_s2a.detach()
        if self.clip_target:
            Q_s2a = F.relu(Q_s2a)
        else:
            q = self.gamma ** q_step * Q_s2a.view(-1, action_num).max(dim=1).values - Q_s1a_

        Q = q.unsqueeze(0)

        path = '/p/reinforcement/data/gnn_rl/model/dqn/dqn_10by10_test_nan/'
        if math.isnan(torch.max(Q)):
            with open(path + 'dqn_nan_debug', 'wb') as model_file:
                pickle.dump([self.model, sample_buffer, Q_s2a.view(-1, action_num).max(dim=1).values, Q_s1a_], model_file)

        return torch.cat(R), Q, sample_buffer

    def back_loss(self, R, Q, update_model=True):
        # print('actual batch size:', R.shape.numel())

        # print('current_R', R)
        # print('current_Q', Q)

        if self.cuda:
            R = R.cuda()

        L = torch.pow(R + Q, 2).sum()
        L.backward(retain_graph=True)


        # print("---------------------------------------------")
        # print(self.model.layers[0].apply_mod.t4.weight.grad.data)
        # print("---------------------------------------------")
        # self.log.add_item('R_signal', R)
        self.Q_err += L.item()

        if update_model:
            self.optimizer.step()

            # # monitor weight change
            # for param in self.model.state_dict().keys():
            #     if param.startswith('MHA'):
            #         continue
            #     pt = self.model.state_dict()[param]
            #     print(param, ':', torch.max(pt).item()
            #                     , torch.min(pt).item()
            #                     , torch.mean(pt).item())

            self.optimizer.zero_grad()
            self.log.add_item('Q_error', self.Q_err)
            self.Q_err = 0
            self.log.add_item('entropy', 0)

    def train_dqn(self, epoch=0, sample_batch_episode=True, batch_size=16, grad_accum=10, num_episodes=10, episode_len=50, gnn_step=10, q_step=1, ddqn=False):
        """
        :param batch_size:
        :param num_episodes:
        :param episode_len: #steps in each episode
        :param gnn_step: #iters when running gnn
        :param q_step: reward delay step
        :param ddqn: train in ddqn mode
        :return:
        """

        print_info = False  # (i % num_episodes == num_episodes - 1)
        if sample_batch_episode:
            T3 = time.time()
            self.run_batch_episode(action_type=self.action_type, gnn_step=gnn_step, episode_len=episode_len,
                                   batch_size=num_episodes, print_info=print_info)
            T4 = time.time()
        else:
            mean_return = 0

            for i in range(num_episodes):
                # 0.2s
                [_, tot_return] = self.run_episode(action_type=self.action_type, gnn_step=gnn_step, episode_len=episode_len, print_info=print_info)
                mean_return = mean_return + tot_return

        # trim experience replay buffer
        self.trim_replay_buffer()

        for i in range(grad_accum):
            # 2s
            T5 = time.time()
            R, Q, sample_buffer = self.sample_from_buffer2(batch_size=batch_size, q_step=q_step, gnn_step=gnn_step)
            # R, Q = self.sample_from_buffer(epoch=epoch, batch_size=batch_size, q_step=q_step, gnn_step=gnn_step, episode_len=episode_len, ddqn=ddqn)
            T6 = time.time()
            # 0.5s
            self.back_loss(R, Q, update_model=(i % grad_accum == grad_accum - 1))
            T7 = time.time()
            del R, Q
            torch.cuda.empty_cache()
        print(T4-T3, T6-T5, T7-T6)
        return self.log, sample_buffer

    def trim_replay_buffer(self):
        if len(self.experience_replay_buffer) > self.replay_buffer_max_size:
            self.experience_replay_buffer = self.experience_replay_buffer[-self.replay_buffer_max_size:]
            self.buffer_indices = self.buffer_indices[-self.replay_buffer_max_size:]
            self.buffer_episode_offset = self.buffer_episode_offset[-1-self.replay_buffer_max_size:]
            shift = self.buffer_episode_offset[0]
            self.buffer_episode_offset = [offset - shift for offset in self.buffer_episode_offset]

        if len(self.experience_replay_buffer2) > self.replay_buffer_max_size2:
            self.experience_replay_buffer2 = self.experience_replay_buffer2[-self.replay_buffer_max_size2:]

    def update_target_net(self):
        # self.model_target = pickle.loads(pickle.dumps(self.model))
        self.model_target.load_state_dict(self.model.state_dict())
