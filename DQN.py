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
import itertools
from toy_models.Qiter import vis_g, state2QtableKey, QtableKey2state


def to_cuda(G_):
    G = dc(G_)
    G.ndata['x'] = G.ndata['x'].cuda()
    G.ndata['label'] = G.ndata['label'].cuda()
    # G.ndata['h'] = G.ndata['h'].cuda()
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
        self.q_pred = []
        self.action_candidates = []
        self.enc_state_seq = []
        self.sub_reward_seq = []
        # self.calib_reward_seq = []
        if self.action_type == 'swap':
            self.label_perm = torch.tensor(range(self.n)).unsqueeze(0)
            self.enc_state_seq.append(state2QtableKey(self.init_state.ndata['label'].argmax(dim=1).cpu().numpy()))
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

    def write(self, action, action_idx, reward, q_val=None, actions=None, state_enc=None, sub_reward=None):


        new_label = self.perm_label(self.label_perm[-1, :], action)

        # new_label_str = ''.join([str(i.item()) for i in new_label[0]])
        # self.visited_state.add(new_label_str)
        self.action_seq.append(action)

        self.action_indices.append(action_idx)

        self.reward_seq.append(reward)

        self.q_pred.append(q_val)

        self.action_candidates.append(actions)

        self.enc_state_seq.append(state_enc)

        self.sub_reward_seq.append(sub_reward)
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
    sub_r: torch.tensor
    s1: DGLGraph
    S: float
    best_S: float
    step: int
    recent_states: list
    forbid_actions: list


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
    def __init__(self, problem
                 , action_type='swap'
                 , gamma=1.0, eps=0.1, lr=1e-4, action_dropout=1.0
                 , sample_batch_episode=False
                 , replay_buffer_max_size=5000
                 , epi_len=50, new_epi_batch_size=10
                 , extended_h=False, time_aware=False, use_x=False
                 , edge_info='adj_weight'
                 , readout='mlp'
                 , explore_method='epsilon_greedy'
                 , priority_sampling='False'
                 , clip_target=False):

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
        self.explore_method = explore_method
        self.clip_target = clip_target
        self.model = DQNet(k=self.k, m=self.m, ajr=self.ajr, num_head=4, hidden_dim=self.hidden_dim, extended_h=self.extended_h, use_x=self.use_x, edge_info=self.edge_info, readout=readout).cuda()
        self.model_target = dc(self.model)
        self.gamma = gamma  # reward decay const
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sample_batch_episode = sample_batch_episode
        self.experience_replay_buffer = []
        self.replay_buffer_max_size = replay_buffer_max_size
        self.buf_epi_len = epi_len  # 50
        self.new_epi_batch_size = new_epi_batch_size  # 10
        self.cascade_replay_buffer = [[] for _ in range(self.buf_epi_len)]
        self.cascade_replay_buffer_weight = torch.zeros((self.buf_epi_len, self.new_epi_batch_size))
        self.stage_max_sizes = [self.replay_buffer_max_size // self.buf_epi_len] * self.buf_epi_len  # [100, 100, ..., 100]
        # self.stage_max_sizes = list(range(100,100+4*50, 4))
        self.buffer_actual_size = sum(self.stage_max_sizes)
        self.priority_sampling = priority_sampling
        self.cascade_buffer_kcut_value = torch.zeros((self.buf_epi_len, self.new_epi_batch_size))
        self.time_aware = time_aware
        self.action_dropout = action_dropout
        self.log = logger()
        self.Q_err = 0  # Q error
        self.log.add_log('tot_return')
        self.log.add_log('Q_error')
        self.log.add_log('entropy')
        self.log.add_log('R_signal')

    def run_batch_episode(self, action_type='swap', gnn_step=3, episode_len=50, batch_size=10):

        sum_r = 0
        bg = to_cuda(self.problem.gen_batch_graph(batch_size=batch_size))
        t = 0

        num_actions = self.problem.get_legal_actions(action_type=action_type, action_dropout=self.action_dropout).shape[0]
        action_mask = torch.tensor(range(0, num_actions * batch_size, num_actions)).cuda()

        explore_dice = (torch.rand(episode_len, batch_size) < self.eps)
        explore_replace_mask = explore_dice.nonzero()  #
        explore_step_offset = torch.cat([torch.zeros([1], dtype=torch.long), torch.cumsum(explore_dice.sum(dim=1), dim=0)], dim=0)
        explore_replace_actions = torch.randint(high=num_actions, size=(explore_replace_mask.shape[0], )).cuda()

        while t < episode_len:

            batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=action_type, action_dropout=self.action_dropout).cuda()

            # epsilon greedy strategy
            # TODO: cannot dc bg once being forwarded.
            # TODO: multi - gpu parallelization
            _, _, _, Q_sa = self.model(dc(bg), batch_legal_actions, action_type=action_type, gnn_step=gnn_step, time_aware=self.time_aware, remain_episode_len=episode_len-t-1)
            best_actions = Q_sa.view(-1, num_actions).argmax(dim=1)

            explore_episode_indices = explore_replace_mask[explore_step_offset[t]: explore_step_offset[t + 1]][:, 1]
            explore_actions = explore_replace_actions[explore_step_offset[t]: explore_step_offset[t + 1]]
            best_actions[explore_episode_indices] = explore_actions
            best_actions += action_mask

            actions = batch_legal_actions[best_actions]

            # update bg inplace and calculate batch rewards
            g0 = [g for g in dgl.unbatch(dc(bg))]  # current_state
            kcut_valueS = self.problem.calc_batchS(bg=bg)
            _, rewards, sub_rewards = self.problem.step_batch(states=bg, action=actions, return_sub_reward=True)
            g1 = [g for g in dgl.unbatch(dc(bg))]  # after_state

            # print('assert=0:', kcut_valueS_ - kcut_valueS + rewards)

            if self.sample_batch_episode:
                self.experience_replay_buffer.extend([sars(g0[i], actions[i], rewards[i], g1[i], kcut_valueS[i], t) for i in range(batch_size)])
            else:  # using cascade buffer
                # update episode best S
                if t > 0:
                    best_S = [np.min([kcut_valueS[i], self.cascade_replay_buffer[t-1][i].best_S]) for i in range(batch_size)]
                else:
                    best_S = [kcut_valueS[i] for i in range(batch_size)]
                self.cascade_replay_buffer[t].extend([sars(g0[i], actions[i], rewards[i], sub_rewards[:, i], g1[i], kcut_valueS[i], best_S[i], t, recent_states=[], forbid_actions=[]) for i in range(batch_size)])
                self.cascade_buffer_kcut_value[t, :batch_size] = kcut_valueS

                if self.priority_sampling:
                    # compute prioritized weights
                    batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=action_type, action_dropout=self.action_dropout).cuda()
                    _, _, _, Q_sa_next = self.model(dc(bg), batch_legal_actions, action_type=action_type, gnn_step=gnn_step,
                                                    time_aware=self.time_aware, remain_episode_len=episode_len - t - 1)
                    delta = Q_sa[best_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)
                    # delta = (Q_sa[best_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)) / (torch.clamp(torch.abs(Q_sa[best_actions]),0.1))
                    self.cascade_replay_buffer_weight[t, :batch_size] = torch.abs(delta.detach())
            R = [reward.item() for reward in rewards]
            sum_r += sum(R)

            t += 1

        self.log.add_item('tot_return', sum_r)

        return R

    def sample_actions_from_q(self, Q_sa, num_actions, batch_size, Temperature=1):

        if self.explore_method == 'epsilon_greedy':

            best_actions = Q_sa.view(-1, num_actions).argmax(dim=1)

            explore_replace_mask = (torch.rand(batch_size) < self.eps).nonzero()
            explore_actions = torch.randint(high=num_actions, size=(explore_replace_mask.shape[0], )).cuda()
            best_actions[explore_replace_mask[:, 0]] = explore_actions

        if self.explore_method == 'softmax' or self.explore_method == 'soft_dqn':
            # print(T)
            best_actions = torch.multinomial(F.softmax(Q_sa.view(-1, num_actions) / Temperature), 1).view(-1)

        # add action batch offset
        best_actions += torch.tensor(range(0, num_actions * batch_size, num_actions)).cuda()
        return best_actions

    def prune_actions(self, bg, recent_states):
        batch_size = bg.batch_size
        n = bg.ndata['label'].shape[0] // batch_size

        batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=self.action_type,
                                                             action_dropout=self.action_dropout).cuda()
        num_actions = batch_legal_actions.shape[0] // batch_size
        forbid_action_mask = torch.zeros(batch_legal_actions.shape[0], 1).cuda()

        for k in range(self.new_epi_batch_size, bg.batch_size):
            cur_state = state2QtableKey(bg.ndata['label'][k*n:(k+1)*n, :].argmax(dim=1).cpu().numpy())  # current state
            forbid_states = set(recent_states[k-self.new_epi_batch_size])
            candicate_actions = batch_legal_actions[k * num_actions:(k + 1) * num_actions, :]
            for j in range(num_actions):
                cur_state_l = QtableKey2state(cur_state)
                cur_state_l[candicate_actions[j][0]] ^= cur_state_l[candicate_actions[j][1]]
                cur_state_l[candicate_actions[j][1]] ^= cur_state_l[candicate_actions[j][0]]
                cur_state_l[candicate_actions[j][0]] ^= cur_state_l[candicate_actions[j][1]]
                if state2QtableKey(cur_state_l) in forbid_states:
                    forbid_action_mask[j + k * num_actions] += 1
        batch_legal_actions = batch_legal_actions * (1 - forbid_action_mask.int()).t().flatten().unsqueeze(1)

        return batch_legal_actions

    def run_cascade_episode(self, action_type='swap', gnn_step=3):

        sum_r = 0
        new_graphs = [to_cuda(self.problem.gen_batch_graph(batch_size=self.new_epi_batch_size))]

        new_graphs.extend(list(itertools.chain(*[[tpl.s1 for tpl in self.cascade_replay_buffer[i][-self.new_epi_batch_size:]] for i in range(self.buf_epi_len-1)])))
        bg = to_cuda(dgl.batch(new_graphs))

        batch_size = self.new_epi_batch_size * self.buf_epi_len
        num_actions = self.problem.get_legal_actions(action_type=action_type, action_dropout=self.action_dropout).shape[0]

        # recent_states = list(itertools.chain(
        #     *[[tpl.recent_states for tpl in self.cascade_replay_buffer[t][-self.new_epi_batch_size:]] for t in
        #       range(self.buf_epi_len - 1)]))
        #
        # batch_legal_actions = self.prune_actions(bg, recent_states)
        batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=self.action_type,
                                                             action_dropout=self.action_dropout).cuda()

        # epsilon greedy strategy
        # TODO: multi-gpu parallelization
        _, _, _, Q_sa = self.model(dc(bg), batch_legal_actions, action_type=action_type, gnn_step=gnn_step)

        # TODO: can alter explore strength according to kcut_valueS
        kcut_valueS = self.problem.calc_batchS(bg=bg)
        chosen_actions = self.sample_actions_from_q(Q_sa, num_actions, batch_size, Temperature=self.eps)
        actions = batch_legal_actions[chosen_actions]

        # update bg inplace and calculate batch rewards
        g0 = [g for g in dgl.unbatch(dc(bg))]  # current_state

        _, rewards, sub_rewards = self.problem.step_batch(states=bg, action=actions, return_sub_reward=True)
        g1 = [g for g in dgl.unbatch(dc(bg))]  # after_state
        # kcut_valueS_ = self.problem.calc_batchS(bg=bg)
        # print('assert=0:', kcut_valueS_ - kcut_valueS + rewards)
        # rewards_ = rewards.reshape(self.buf_epi_len, -1)
        # posi_r = rewards_ >= 0
        # nege_r = rewards_ < 0
        # scaled_rewards = (rewards_-torch.mean(rewards_, dim=1).t().unsqueeze(1)) / torch.std(rewards_, dim=1).t().unsqueeze(1)
        # scaled_rewards = scaled_rewards * posi_r + rewards_ * nege_r
        #
        # scaled_rewards = scaled_rewards.view(-1)

        # update episode best S

        best_S = torch.cat([kcut_valueS[:self.new_epi_batch_size].cpu() \
                    , torch.tensor([[np.min([kcut_valueS[j+t*self.new_epi_batch_size], self.cascade_replay_buffer[t - 1][j-self.new_epi_batch_size].best_S]) for j in range(self.new_epi_batch_size)] for t in range(1, self.buf_epi_len)]).flatten()])

        # update recent states
        recent_states = [[state2QtableKey(g.ndata['label'].argmax(dim=1).cpu().numpy())] for g in g0[:self.new_epi_batch_size]]
        recent_states.extend(
            list(itertools.chain(
            *[
                [
                    self.cascade_replay_buffer[t - 1][j - self.new_epi_batch_size].recent_states
                    + [
                        state2QtableKey(g0[j+t*self.new_epi_batch_size].ndata['label'].argmax(dim=1).cpu().numpy())
                       ]
                 for j in range(self.new_epi_batch_size)
                 ] for t in range(1, self.buf_epi_len)
            ]))
        )

        memory_len = 10
        recent_states = [state[-memory_len:] for state in recent_states]
        forbid_actions = []

        [self.cascade_replay_buffer[t].extend(
            [sars(g0[j+t*self.new_epi_batch_size]
            , actions[j+t*self.new_epi_batch_size]
            , rewards[j+t*self.new_epi_batch_size]
            , sub_rewards[:, j + t * self.new_epi_batch_size]
            , g1[j+t*self.new_epi_batch_size]
            , kcut_valueS[j+t*self.new_epi_batch_size]
            , best_S[j+t*self.new_epi_batch_size]
            , t
            , recent_states[j+t*self.new_epi_batch_size]
            , forbid_actions)
            for j in range(self.new_epi_batch_size)])
         for t in range(self.buf_epi_len)]

        self.cascade_buffer_kcut_value = torch.cat([self.cascade_buffer_kcut_value, torch.abs(kcut_valueS.detach().cpu().view(self.buf_epi_len, self.new_epi_batch_size))], dim=1).detach()

        if self.priority_sampling:
            # compute prioritized weights
            batch_legal_actions = self.problem.get_legal_actions(state=bg, action_type=action_type, action_dropout=self.action_dropout).cuda()
            _, _, _, Q_sa_next = self.model(dc(bg), batch_legal_actions, action_type=action_type, gnn_step=gnn_step)

            delta = Q_sa[chosen_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)
            # delta = (Q_sa[chosen_actions] - (rewards + self.gamma * Q_sa_next.view(-1, num_actions).max(dim=1).values)) / torch.clamp(torch.abs(Q_sa[chosen_actions]),0.1)
            self.cascade_replay_buffer_weight = torch.cat([self.cascade_replay_buffer_weight, torch.abs(delta.detach().cpu().view(self.buf_epi_len, self.new_epi_batch_size))], dim=1).detach()
            # print(self.cascade_replay_buffer_weight)
        R = [reward.item() for reward in rewards]
        sum_r += sum(R)

        self.log.add_item('tot_return', sum_r)

        return R

    def sample_from_buffer(self, batch_size, q_step, gnn_step):

        batch_size = min(batch_size, len(self.experience_replay_buffer))

        sample_buffer = np.random.choice(self.experience_replay_buffer, batch_size, replace=False)
        # make batches
        batch_begin_state = dgl.batch([tpl.s0 for tpl in sample_buffer])
        batch_end_state = dgl.batch([tpl.s1 for tpl in sample_buffer])
        R = [tpl.r.unsqueeze(0) for tpl in sample_buffer]
        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)
        batch_end_action = self.problem.get_legal_actions(state=batch_end_state, action_type=self.action_type, action_dropout=self.action_dropout).cuda()
        action_num = batch_end_action.shape[0] // batch_begin_action.shape[0]

        # only compute limited number for Q_s1a
        # TODO: multi-gpu parallelization
        _, _, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step)
        _, _, _, Q_s2a = self.model_target(batch_end_state, batch_end_action, action_type=self.action_type, gnn_step=gnn_step)
        # Q_s2a = Q_s2a.detach()
        if self.clip_target:
            Q_s2a = F.relu(Q_s2a)
        else:
            q = self.gamma ** q_step * Q_s2a.view(-1, action_num).max(dim=1).values - Q_s1a_

        Q = q.unsqueeze(0)

        return torch.cat(R), Q, sample_buffer

    def sample_from_cascade_buffer(self, batch_size, q_step, gnn_step):

        batch_size = min(batch_size, len(self.cascade_replay_buffer[0]) * self.buf_epi_len)

        if self.priority_sampling:
            nc = self.cascade_replay_buffer_weight.shape[1]
            selected_indices = torch.multinomial((0.1 + self.cascade_replay_buffer_weight.flatten()) ** 1.0, batch_size)
            sample_buffer = [self.cascade_replay_buffer[indices // nc][indices % nc] for indices in selected_indices]

        else:
            if False:
                # importance sampling according to S
                nc = self.cascade_buffer_kcut_value.shape[1]
                print('sample weight:', torch.sum(1.0 / self.cascade_buffer_kcut_value ** 2, dim=1))
                selected_indices = torch.multinomial((1.0 / self.cascade_buffer_kcut_value.flatten() ** 2), batch_size)
                sample_buffer = [self.cascade_replay_buffer[indices // nc][indices % nc] for indices in
                                 selected_indices]
            else:
                batch_sizes = [
                    min(batch_size * self.stage_max_sizes[i] // self.buffer_actual_size, len(self.cascade_replay_buffer[0]))
                    for i in range(self.buf_epi_len)]
                sample_buffer = list(itertools.chain(*[np.random.choice(a=self.cascade_replay_buffer[i]
                                                                    , size=batch_sizes[i]
                                                                    , replace=False
                                                                    # , p=
                                                                    ) for i in range(self.buf_epi_len)]))

        # make batches
        batch_begin_state = dgl.batch([tpl.s0 for tpl in sample_buffer])
        batch_end_state = dgl.batch([tpl.s1 for tpl in sample_buffer])
        R = [tpl.r.unsqueeze(0) for tpl in sample_buffer]
        sub_R = torch.cat([tpl.sub_r.unsqueeze(0) for tpl in sample_buffer])  # (b, k)

        batch_begin_action = torch.cat([tpl.a.unsqueeze(0) for tpl in sample_buffer], axis=0)

        batch_end_action = self.problem.get_legal_actions(state=batch_end_state, action_type=self.action_type, action_dropout=self.action_dropout).cuda()
        action_num = batch_end_action.shape[0] // batch_begin_action.shape[0]

        # only compute limited number for Q_s1a
        # TODO: multi-gpu parallelization
        _, _, _, Q_s1a_ = self.model(batch_begin_state, batch_begin_action, action_type=self.action_type, gnn_step=gnn_step)
        _, _, _, Q_s2a = self.model_target(batch_end_state, batch_end_action, action_type=self.action_type, gnn_step=gnn_step)

        # reward shaping
        V_s = Q_s2a.view(-1, action_num).sum(dim=1)
        beta = 0.1 / (torch.abs(V_s) + 0.1)
        #  when close to opt, reward negative sub_reward
        internal_R = beta * F.relu(-sub_R).sum(dim=1)

        if self.clip_target:
            Q_s2a = F.relu(Q_s2a)

        if self.explore_method == 'soft_dqn':
            sample_prob = F.softmax(Q_s2a.view(-1, action_num) / self.eps)
            q = self.gamma ** q_step * torch.sum(sample_prob * Q_s2a.view(-1, action_num), dim=1) - Q_s1a_
        else:
            chosen_actions = self.sample_actions_from_q(Q_s2a, action_num, batch_size, Temperature=self.eps)
            q = self.gamma ** q_step * Q_s2a[chosen_actions] - Q_s1a_

        # q = self.gamma ** q_step * Q_s2a.view(-1, action_num).max(dim=1).values - Q_s1a_
        if self.priority_sampling:
            # print(self.cascade_replay_buffer_weight.view(-1)[selected_indices])
            self.cascade_replay_buffer_weight.flatten()[selected_indices] = torch.abs(q).cpu()
            # self.cascade_replay_buffer_weight.flatten()[selected_indices] = torch.abs(q / torch.clamp(torch.abs(Q_s1a_),0.1)).cpu()
            print(self.cascade_replay_buffer_weight.flatten()[selected_indices])

        Q = q.unsqueeze(0)

        return torch.cat(R), internal_R, Q, sample_buffer

    def back_loss(self, R, Q, update_model=True):

        if self.clip_target:
            L = torch.pow(F.relu(R.cuda()) + Q, 2).sum()
        else:
            L = torch.pow(R.cuda() + Q, 2).sum()
        L.backward(retain_graph=True)

        self.Q_err += L.item()

        if update_model:
            self.optimizer.step()

            self.optimizer.zero_grad()
            self.log.add_item('Q_error', self.Q_err)
            self.Q_err = 0
            self.log.add_item('entropy', 0)

    def train_dqn(self, epoch=0, batch_size=16, num_episodes=10, episode_len=50, gnn_step=10, q_step=1, ddqn=False):
        """
        :param batch_size:
        :param num_episodes:
        :param episode_len: #steps in each episode
        :param gnn_step: #iters when running gnn
        :param q_step: reward delay step
        :param ddqn: train in ddqn mode
        :return:
        """
        if self.sample_batch_episode:
            T3 = time.time()
            self.run_batch_episode(action_type=self.action_type, gnn_step=gnn_step, episode_len=episode_len,
                                   batch_size=num_episodes)
            T4 = time.time()

            # trim experience replay buffer
            self.trim_replay_buffer(epoch)

            R, Q, sample_buffer = self.sample_from_buffer(batch_size=batch_size, q_step=q_step, gnn_step=gnn_step)

            T6 = time.time()
        else:
            T3 = time.time()
            if epoch == 0:
                self.run_batch_episode(action_type=self.action_type, gnn_step=gnn_step, episode_len=self.buf_epi_len,
                                   batch_size=self.new_epi_batch_size)
                # change step batch mask helper
                self.problem.sample_episode = self.buf_epi_len * self.new_epi_batch_size
                self.problem.gen_step_batch_mask()
            else:
                self.run_cascade_episode(action_type=self.action_type, gnn_step=gnn_step)
            T4 = time.time()
            # trim experience replay buffer
            self.trim_replay_buffer(epoch)

            R, internal_R, Q, sample_buffer = self.sample_from_cascade_buffer(batch_size=batch_size, q_step=q_step, gnn_step=gnn_step)
            print('external R:', R)
            print('internal_R:', internal_R)
            T6 = time.time()
        # 0.5s
        self.back_loss(R, Q, update_model=True)
        T7 = time.time()
        del R, Q
        torch.cuda.empty_cache()

        print(T4-T3, T6-T4, T7-T6)

        return self.log, sample_buffer

    def trim_replay_buffer(self, epoch):
        if len(self.experience_replay_buffer) > self.replay_buffer_max_size:
            self.experience_replay_buffer = self.experience_replay_buffer[-self.replay_buffer_max_size:]

        if epoch * self.buf_epi_len * self.new_epi_batch_size > self.replay_buffer_max_size:
            for i in range(self.buf_epi_len):
                self.cascade_replay_buffer[i] = self.cascade_replay_buffer[i][-self.stage_max_sizes[i]:]
        self.cascade_replay_buffer_weight = self.cascade_replay_buffer_weight[:, -self.stage_max_sizes[0]:]
        self.cascade_buffer_kcut_value = self.cascade_buffer_kcut_value[:, -self.stage_max_sizes[0]:]


    def update_target_net(self):
        self.model_target.load_state_dict(self.model.state_dict())
